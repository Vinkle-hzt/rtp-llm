from typing import Any

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.base_rotary_embedding_op import (
    BaseRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs


class MhaRotaryEmbeddingOp(BaseRotaryEmbeddingOp):
    """Rotary positional embedding for Multi-Head Attention (MHA)."""

    def __init__(
        self,
        attn_config: AttentionConfigs,
        cos_sin_cache: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize MHA Rotary Embedding operator.

        Note: This op only applies RoPE. For KV cache writing, use KVCacheWriteOp separately.

        Args:
            attn_config: Attention configuration containing all necessary parameters
            cos_sin_cache: Precomputed cos/sin cache for RoPE, shape [max_seq_len, head_dim].
                          Layout: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
                          where d = head_dim. First half stores cosine values, second half stores sine values.
                          dtype should be torch.float32 for numerical stability.
                          If None, will auto-generate using attn_config.rope_config.
        """
        super().__init__(
            attn_config.size_per_head,
            cos_sin_cache,
            attn_config.kernel_tokens_per_block,
            is_neox_style=attn_config.rope_config.is_neox_style,
            rope_config=attn_config.rope_config,
            max_position_embeddings=attn_config.max_seq_len
            + attn_config.gen_num_per_cycle
            + 1,
        )
        self.num_heads = attn_config.head_num
        self.num_kv_heads = attn_config.kv_head_num
        self.seq_size_per_block = attn_config.kernel_tokens_per_block
        self.params = None

    def set_params(self, params: Any):
        """Set the params object to be filled by this op."""
        self.params = params

    def forward(  # type: ignore
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to QKV tensor for MHA.

        Note: This op only applies RoPE. KV cache writing should be done by KVCacheWriteOp separately.

        Args:
            qkv: QKV tensor [total_tokens, hidden_size] where hidden_size = (num_heads + 2*num_kv_heads) * head_dim

        Returns:
            Tuple of (query, key, value) tensors after RoPE:
                - query: [total_tokens, num_heads, head_dim]
                - key: [total_tokens, num_kv_heads, head_dim]
                - value: [total_tokens, num_kv_heads, head_dim]
        """
        # Split QKV tensor into Q, K, V
        # qkv shape: [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_kv_heads,
                self.head_size * self.num_kv_heads,
            ],
            dim=-1,
        )
        # Reshape to [total_tokens, num_heads, head_dim]
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size)
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size)

        # Apply RoPE to Q and K
        self._apply_rope(query, key, self.params)

        return query, key, value


class MropeRotaryEmbeddingOp:
    """Device-only interleaved MRoPE used by Qwen3.5 target verification."""

    def __init__(self, attn_config: AttentionConfigs) -> None:
        self.head_size = attn_config.size_per_head
        self.num_heads = attn_config.head_num
        self.num_kv_heads = attn_config.kv_head_num
        self.rope_dim = attn_config.rope_config.dim
        self.index_factor = attn_config.rope_config.index_factor
        self.inv_freq = 1.0 / torch.pow(
            float(attn_config.rope_config.base),
            torch.arange(0, self.rope_dim, 2, device="cuda").float()
            / self.rope_dim,
        )
        remaining = [
            attn_config.rope_config.mrope_dim1,
            attn_config.rope_config.mrope_dim2,
            attn_config.rope_config.mrope_dim3,
        ]
        axis_indices = []
        while any(remaining):
            for axis in range(3):
                if remaining[axis] > 0:
                    axis_indices.append(axis)
                    remaining[axis] -= 1
        self.axis_indices = torch.tensor(axis_indices, dtype=torch.long, device="cuda")
        assert self.index_factor == 3
        assert self.axis_indices.numel() == self.rope_dim // 2

    def forward(
        self,
        qkv: torch.Tensor,
        position_ids: torch.Tensor,
        fallback_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        query, key, value = torch.split(
            qkv, [q_size, kv_size, kv_size], dim=-1
        )
        query = query.view(qkv.shape[0], self.num_heads, self.head_size)
        key = key.view(qkv.shape[0], self.num_kv_heads, self.head_size)
        value = value.view(qkv.shape[0], self.num_kv_heads, self.head_size)

        positions = position_ids.narrow(
            0, 0, qkv.shape[0] * self.index_factor
        ).view(qkv.shape[0], self.index_factor)
        fallback = fallback_positions.narrow(0, 0, qkv.shape[0]).unsqueeze(1)
        positions = torch.where(positions > 0, positions, fallback)
        selected_positions = positions[:, self.axis_indices].float()
        angles = selected_positions * self.inv_freq.unsqueeze(0)
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        def apply(tensor: torch.Tensor) -> None:
            rope = tensor[..., : self.rope_dim]
            first, second = rope.chunk(2, dim=-1)
            rotated = torch.cat([-second, first], dim=-1)
            rope.copy_(
                rope * torch.cat([cos, cos], dim=-1)
                + rotated * torch.cat([sin, sin], dim=-1)
            )

        apply(query)
        apply(key)
        return query, key, value
