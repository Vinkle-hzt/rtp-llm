#include "rtp_llm/models_py/bindings/common/kernels/mtp_target_verify_prepare.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>
#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

namespace {

__global__ void mtpTargetVerifyPrepareKernel(const int32_t* __restrict__ sequence_lengths,
                                             int32_t* __restrict__ input_lengths,
                                             int32_t* __restrict__ prefix_lengths,
                                             int32_t* __restrict__ sequence_lengths_plus_1,
                                             int32_t* __restrict__ lm_output_indexes,
                                             int32_t tokens_per_batch,
                                             int32_t batch_size) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size) {
        return;
    }
    input_lengths[idx]           = tokens_per_batch;
    prefix_lengths[idx]          = sequence_lengths[idx];
    sequence_lengths_plus_1[idx] = sequence_lengths[idx] + 1;
    lm_output_indexes[idx]       = idx * tokens_per_batch;
}

__global__ void mtpSpecDecodeMetadataPrepareKernel(int32_t* __restrict__ input_lengths,
                                                   int32_t* __restrict__ lm_output_indexes,
                                                   int32_t tokens_per_batch,
                                                   int32_t batch_size) {
    const int32_t total_tokens = batch_size * tokens_per_batch;
    const int32_t idx          = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < batch_size) {
        input_lengths[idx] = tokens_per_batch;
    }
    if (idx < total_tokens) {
        lm_output_indexes[idx] = idx;
    }
}

__global__ void mtpSpecDecodeTokensMetadataPrepareKernel(const int32_t* __restrict__ token0,
                                                         const int32_t* __restrict__ token1,
                                                         const int32_t* __restrict__ token2,
                                                         const int32_t* __restrict__ token3,
                                                         const int32_t* __restrict__ token4,
                                                         const int32_t* __restrict__ token5,
                                                         const int32_t* __restrict__ token6,
                                                         const int32_t* __restrict__ token7,
                                                         int32_t* __restrict__ spec_tokens,
                                                         int32_t* __restrict__ input_lengths,
                                                         int32_t* __restrict__ lm_output_indexes,
                                                         int32_t tokens_per_batch,
                                                         int32_t batch_size) {
    const int32_t total_tokens = batch_size * tokens_per_batch;
    const int32_t idx          = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_tokens) {
        return;
    }

    const int32_t  batch_idx = idx / tokens_per_batch;
    const int32_t  token_idx = idx - batch_idx * tokens_per_batch;
    const int32_t* src       = nullptr;
    switch (token_idx) {
        case 0:
            src = token0;
            break;
        case 1:
            src = token1;
            break;
        case 2:
            src = token2;
            break;
        case 3:
            src = token3;
            break;
        case 4:
            src = token4;
            break;
        case 5:
            src = token5;
            break;
        case 6:
            src = token6;
            break;
        case 7:
            src = token7;
            break;
    }

    spec_tokens[idx]       = src[batch_idx];
    lm_output_indexes[idx] = idx;
    if (token_idx == 0) {
        input_lengths[batch_idx] = tokens_per_batch;
    }
}

void checkCudaI32Vector(const torch::Tensor& tensor, const char* name, int64_t batch_size) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(
        tensor.numel() >= batch_size, "%s numel %ld is smaller than batch_size %ld", name, tensor.numel(), batch_size);
}

void checkCudaI32Tensor(const torch::Tensor& tensor, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be CUDA", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "%s must be int32", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
}

__device__ __forceinline__ int32_t readPhysicalBlockId(const int32_t* block_row,
                                                       int32_t        physical_pos,
                                                       int32_t        kernel_blocks_per_kv_block) {
    const int32_t kernel_id = block_row[physical_pos * kernel_blocks_per_kv_block];
    return kernel_id < 0 ? -1 : kernel_id / kernel_blocks_per_kv_block;
}

__device__ __forceinline__ int32_t readPatchedPhysicalBlockId(const int32_t* block_row,
                                                              int32_t        physical_pos,
                                                              int32_t        kernel_blocks_per_kv_block,
                                                              const int32_t* positions,
                                                              const int32_t* values,
                                                              int32_t        count) {
    for (int32_t i = 0; i < count; ++i) {
        if (positions[i] == physical_pos) {
            return values[i];
        }
    }
    return readPhysicalBlockId(block_row, physical_pos, kernel_blocks_per_kv_block);
}

__device__ __forceinline__ void recordPhysicalBlockAssignment(
    int32_t physical_pos, int32_t physical_id, int32_t* positions, int32_t* values, int32_t& count) {
    for (int32_t i = 0; i < count; ++i) {
        if (positions[i] == physical_pos) {
            values[i] = physical_id;
            return;
        }
    }
    positions[count] = physical_pos;
    values[count]    = physical_id;
    ++count;
}

__device__ __forceinline__ void simulatePhysicalBlockSwap(const int32_t* block_row,
                                                          int32_t        rhs,
                                                          int32_t        lhs,
                                                          int32_t        kernel_blocks_per_kv_block,
                                                          int32_t*       positions,
                                                          int32_t*       values,
                                                          int32_t&       count) {
    if (rhs == lhs) {
        return;
    }
    const int32_t rhs_value =
        readPatchedPhysicalBlockId(block_row, rhs, kernel_blocks_per_kv_block, positions, values, count);
    const int32_t lhs_value =
        readPatchedPhysicalBlockId(block_row, lhs, kernel_blocks_per_kv_block, positions, values, count);
    recordPhysicalBlockAssignment(rhs, lhs_value, positions, values, count);
    recordPhysicalBlockAssignment(lhs, rhs_value, positions, values, count);
}

}  // namespace

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
                                  int32_t              tokens_per_batch,
                                  MtpPrepareStream     stream) {
    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(sequence_lengths, "sequence_lengths", batch_size);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(prefix_lengths, "prefix_lengths", batch_size);
    checkCudaI32Vector(sequence_lengths_plus_1, "sequence_lengths_plus_1", batch_size);
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", batch_size);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpTargetVerifyPrepareKernel<<<grid_size, block_size, 0, stream>>>(sequence_lengths.data_ptr<int32_t>(),
                                                                       input_lengths.data_ptr<int32_t>(),
                                                                       prefix_lengths.data_ptr<int32_t>(),
                                                                       sequence_lengths_plus_1.data_ptr<int32_t>(),
                                                                       lm_output_indexes.data_ptr<int32_t>(),
                                                                       tokens_per_batch,
                                                                       static_cast<int32_t>(batch_size));
}

void invokeMtpSpecDecodeMetadataPrepare(torch::Tensor&   input_lengths,
                                        torch::Tensor&   lm_output_indexes,
                                        int32_t          tokens_per_batch,
                                        MtpPrepareStream stream) {
    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    const int64_t total_tokens = batch_size * tokens_per_batch;
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", total_tokens);

    constexpr int block_size = 256;
    const int64_t work_items = std::max<int64_t>(batch_size, total_tokens);
    const int     grid_size  = static_cast<int>((work_items + block_size - 1) / block_size);
    mtpSpecDecodeMetadataPrepareKernel<<<grid_size, block_size, 0, stream>>>(input_lengths.data_ptr<int32_t>(),
                                                                             lm_output_indexes.data_ptr<int32_t>(),
                                                                             tokens_per_batch,
                                                                             static_cast<int32_t>(batch_size));
}

void invokeMtpSpecDecodeTokensMetadataPrepare(const std::vector<torch::Tensor>& token_columns,
                                              torch::Tensor&                    spec_tokens,
                                              torch::Tensor&                    input_lengths,
                                              torch::Tensor&                    lm_output_indexes,
                                              int32_t                           tokens_per_batch,
                                              MtpPrepareStream                  stream) {
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch > 0, "tokens_per_batch must be positive");
    RTP_LLM_CHECK_WITH_INFO(tokens_per_batch <= 8, "tokens_per_batch %d exceeds fused kernel max 8", tokens_per_batch);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int32_t>(token_columns.size()) == tokens_per_batch,
                            "token_columns size %ld must equal tokens_per_batch %d",
                            token_columns.size(),
                            tokens_per_batch);

    const int64_t batch_size = input_lengths.numel();
    if (batch_size <= 0) {
        return;
    }
    const int64_t total_tokens = batch_size * tokens_per_batch;
    checkCudaI32Vector(spec_tokens, "spec_tokens", total_tokens);
    checkCudaI32Vector(input_lengths, "input_lengths", batch_size);
    checkCudaI32Vector(lm_output_indexes, "lm_output_indexes", total_tokens);
    for (size_t i = 0; i < token_columns.size(); ++i) {
        checkCudaI32Vector(token_columns[i], "token_columns", batch_size);
    }

    const int32_t* ptrs[8] = {};
    for (size_t i = 0; i < token_columns.size(); ++i) {
        ptrs[i] = token_columns[i].data_ptr<int32_t>();
    }

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((total_tokens + block_size - 1) / block_size);
    mtpSpecDecodeTokensMetadataPrepareKernel<<<grid_size, block_size, 0, stream>>>(
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        spec_tokens.data_ptr<int32_t>(),
        input_lengths.data_ptr<int32_t>(),
        lm_output_indexes.data_ptr<int32_t>(),
        tokens_per_batch,
        static_cast<int32_t>(batch_size));
}

// Fused kernel: next_seq_len[i] = prev_seq_len[i] + accept_len[i]
//               hidden_idx[i]  = (int64_t)(accept_len[i] - 1)
__global__ void mtpDispatchStatePrepareKernel(const int32_t* __restrict__ accept_len,
                                              const int32_t* __restrict__ prev_seq_len,
                                              int32_t* __restrict__ next_seq_len,
                                              int64_t* __restrict__ hidden_idx,
                                              int32_t batch_size) {
    const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= batch_size) {
        return;
    }
    const int32_t al  = accept_len[idx];
    next_seq_len[idx] = prev_seq_len[idx] + al;
    hidden_idx[idx]   = static_cast<int64_t>(al - 1);
}

void invokeMtpDispatchStatePrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& prev_seq_len,
                                   torch::Tensor&       next_seq_len,
                                   torch::Tensor&       hidden_idx,
                                   int64_t              batch_size,
                                   MtpPrepareStream     stream) {
    if (batch_size <= 0) {
        return;
    }
    checkCudaI32Vector(accept_len, "accept_len", batch_size);
    checkCudaI32Vector(prev_seq_len, "prev_seq_len", batch_size);
    checkCudaI32Vector(next_seq_len, "next_seq_len", batch_size);
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.defined() && hidden_idx.is_cuda(), "hidden_idx must be CUDA");
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.scalar_type() == torch::kInt64, "hidden_idx must be int64");
    RTP_LLM_CHECK_WITH_INFO(hidden_idx.is_contiguous(), "hidden_idx must be contiguous");
    RTP_LLM_CHECK_WITH_INFO(
        hidden_idx.numel() >= batch_size, "hidden_idx numel %ld < batch_size %ld", hidden_idx.numel(), batch_size);

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>((batch_size + block_size - 1) / block_size);
    mtpDispatchStatePrepareKernel<<<grid_size, block_size, 0, stream>>>(accept_len.data_ptr<int32_t>(),
                                                                        prev_seq_len.data_ptr<int32_t>(),
                                                                        next_seq_len.data_ptr<int32_t>(),
                                                                        hidden_idx.data_ptr<int64_t>(),
                                                                        static_cast<int32_t>(batch_size));
}

__global__ void mtpBuildLinearBlockUpdatesKernel(const int32_t* __restrict__ accept_len,
                                                 const int32_t* __restrict__ prev_seq_len,
                                                 const int32_t* __restrict__ block_table,
                                                 const int32_t* __restrict__ linear_group_ids,
                                                 int32_t* __restrict__ updates,
                                                 int32_t batch_size,
                                                 int32_t group_count,
                                                 int32_t linear_group_count,
                                                 int32_t kernel_block_count,
                                                 int32_t seq_size_per_block,
                                                 int32_t kernel_blocks_per_kv_block) {
    const int32_t work_idx  = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t work_size = batch_size * linear_group_count;
    if (work_idx >= work_size) {
        return;
    }

    const int32_t batch_idx        = work_idx / linear_group_count;
    const int32_t linear_group_idx = work_idx - batch_idx * linear_group_count;
    int32_t*      output           = updates + work_idx * 8;
    for (int32_t i = 0; i < 4; ++i) {
        output[i * 2]     = -1;
        output[i * 2 + 1] = -1;
    }

    const int32_t group_id = linear_group_ids[linear_group_idx];
    if (group_id < 0 || group_id >= group_count || accept_len[batch_idx] <= 1 || prev_seq_len[batch_idx] <= 0) {
        return;
    }

    const int32_t physical_block_count = kernel_block_count / kernel_blocks_per_kv_block;
    const int32_t cur_cached_len       = prev_seq_len[batch_idx] - 1;
    const int32_t nxt_cached_len       = cur_cached_len + accept_len[batch_idx];

    int32_t cached_src_block_idx = 0;
    int32_t cached_des_block_idx = 0;
    if ((cur_cached_len + 1) % seq_size_per_block > (nxt_cached_len + seq_size_per_block - 1) % seq_size_per_block) {
        const int32_t base_block_idx      = cur_cached_len / seq_size_per_block;
        const int32_t cached_token_offset = seq_size_per_block - cur_cached_len % seq_size_per_block - 1;
        cached_src_block_idx              = base_block_idx + cached_token_offset;
        cached_des_block_idx              = nxt_cached_len / seq_size_per_block - 1;
    }

    const int32_t base_block_idx = cur_cached_len / seq_size_per_block;
    const int32_t src_block_idx  = base_block_idx + accept_len[batch_idx] - 1;
    const int32_t des_block_idx  = (nxt_cached_len - 1) / seq_size_per_block;
    if (cached_src_block_idx < 0 || cached_src_block_idx >= physical_block_count || cached_des_block_idx < 0
        || cached_des_block_idx >= physical_block_count || src_block_idx < 0 || src_block_idx >= physical_block_count
        || des_block_idx < 0 || des_block_idx >= physical_block_count) {
        return;
    }

    const int32_t* block_row = block_table + (group_id * batch_size + batch_idx) * kernel_block_count;
    int32_t        positions[4];
    int32_t        values[4];
    int32_t        count = 0;
    simulatePhysicalBlockSwap(
        block_row, cached_src_block_idx, cached_des_block_idx, kernel_blocks_per_kv_block, positions, values, count);
    simulatePhysicalBlockSwap(
        block_row, src_block_idx, des_block_idx, kernel_blocks_per_kv_block, positions, values, count);

    for (int32_t i = 0; i < count; ++i) {
        output[i * 2]     = positions[i];
        output[i * 2 + 1] = values[i];
    }
}

__global__ void mtpApplyLinearBlockUpdatesKernel(int32_t* __restrict__ block_table,
                                                 const int32_t* __restrict__ linear_group_ids,
                                                 const int32_t* __restrict__ updates,
                                                 int32_t batch_size,
                                                 int32_t group_count,
                                                 int32_t linear_group_count,
                                                 int32_t kernel_block_count,
                                                 int32_t kernel_blocks_per_kv_block) {
    const int32_t work_idx  = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t work_size = batch_size * linear_group_count;
    if (work_idx >= work_size) {
        return;
    }

    const int32_t batch_idx        = work_idx / linear_group_count;
    const int32_t linear_group_idx = work_idx - batch_idx * linear_group_count;
    const int32_t group_id         = linear_group_ids[linear_group_idx];
    if (group_id < 0 || group_id >= group_count) {
        return;
    }

    const int32_t  physical_block_count = kernel_block_count / kernel_blocks_per_kv_block;
    int32_t*       block_row            = block_table + (group_id * batch_size + batch_idx) * kernel_block_count;
    const int32_t* input                = updates + work_idx * 8;
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t physical_pos = input[i * 2];
        const int32_t physical_id  = input[i * 2 + 1];
        if (physical_pos < 0) {
            continue;
        }
        if (physical_pos >= physical_block_count) {
            continue;
        }
        const int32_t kernel_pos = physical_pos * kernel_blocks_per_kv_block;
        for (int32_t j = 0; j < kernel_blocks_per_kv_block; ++j) {
            block_row[kernel_pos + j] = physical_id < 0 ? -1 : physical_id * kernel_blocks_per_kv_block + j;
        }
    }
}

void invokeMtpBuildLinearBlockUpdates(const torch::Tensor& accept_len,
                                      const torch::Tensor& prev_seq_len,
                                      const torch::Tensor& kv_cache_kernel_block_id,
                                      const torch::Tensor& linear_group_ids,
                                      torch::Tensor&       linear_block_updates,
                                      int32_t              seq_size_per_block,
                                      int32_t              kernel_blocks_per_kv_block,
                                      MtpPrepareStream     stream) {
    checkCudaI32Tensor(kv_cache_kernel_block_id, "kv_cache_kernel_block_id");
    checkCudaI32Tensor(linear_group_ids, "linear_group_ids");
    checkCudaI32Tensor(linear_block_updates, "linear_block_updates");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.dim() == 3, "kv_cache_kernel_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(linear_block_updates.dim() == 4 && linear_block_updates.size(2) == 4
                                && linear_block_updates.size(3) == 2,
                            "linear_block_updates must have shape [batch, linear_group_num, 4, 2]");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be positive");
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_kv_block > 0, "kernel_blocks_per_kv_block must be positive");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.size(2) % kernel_blocks_per_kv_block == 0,
                            "kernel block count must be divisible by kernel_blocks_per_kv_block");

    const int64_t batch_size       = linear_block_updates.size(0);
    const int64_t linear_group_num = linear_block_updates.size(1);
    checkCudaI32Vector(accept_len, "accept_len", batch_size);
    checkCudaI32Vector(prev_seq_len, "prev_seq_len", batch_size);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.size(1) == batch_size,
                            "block-table batch %ld != update batch %ld",
                            kv_cache_kernel_block_id.size(1),
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(linear_group_ids.numel() == linear_group_num,
                            "linear_group_ids numel %ld != update group count %ld",
                            linear_group_ids.numel(),
                            linear_group_num);
    if (batch_size == 0 || linear_group_num == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int64_t work_size  = batch_size * linear_group_num;
    const int     grid_size  = static_cast<int>((work_size + block_size - 1) / block_size);
    mtpBuildLinearBlockUpdatesKernel<<<grid_size, block_size, 0, stream>>>(
        accept_len.data_ptr<int32_t>(),
        prev_seq_len.data_ptr<int32_t>(),
        kv_cache_kernel_block_id.data_ptr<int32_t>(),
        linear_group_ids.data_ptr<int32_t>(),
        linear_block_updates.data_ptr<int32_t>(),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(kv_cache_kernel_block_id.size(0)),
        static_cast<int32_t>(linear_group_num),
        static_cast<int32_t>(kv_cache_kernel_block_id.size(2)),
        seq_size_per_block,
        kernel_blocks_per_kv_block);
}

void invokeMtpApplyLinearBlockUpdates(torch::Tensor&       kv_cache_kernel_block_id,
                                      const torch::Tensor& linear_group_ids,
                                      const torch::Tensor& linear_block_updates,
                                      int32_t              kernel_blocks_per_kv_block,
                                      MtpPrepareStream     stream) {
    checkCudaI32Tensor(kv_cache_kernel_block_id, "kv_cache_kernel_block_id");
    checkCudaI32Tensor(linear_group_ids, "linear_group_ids");
    checkCudaI32Tensor(linear_block_updates, "linear_block_updates");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.dim() == 3, "kv_cache_kernel_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(linear_block_updates.dim() == 4 && linear_block_updates.size(2) == 4
                                && linear_block_updates.size(3) == 2,
                            "linear_block_updates must have shape [batch, linear_group_num, 4, 2]");
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_kv_block > 0, "kernel_blocks_per_kv_block must be positive");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.size(2) % kernel_blocks_per_kv_block == 0,
                            "kernel block count must be divisible by kernel_blocks_per_kv_block");

    const int64_t batch_size       = linear_block_updates.size(0);
    const int64_t linear_group_num = linear_block_updates.size(1);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id.size(1) == batch_size,
                            "block-table batch %ld != update batch %ld",
                            kv_cache_kernel_block_id.size(1),
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(linear_group_ids.numel() == linear_group_num,
                            "linear_group_ids numel %ld != update group count %ld",
                            linear_group_ids.numel(),
                            linear_group_num);
    if (batch_size == 0 || linear_group_num == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int64_t work_size  = batch_size * linear_group_num;
    const int     grid_size  = static_cast<int>((work_size + block_size - 1) / block_size);
    mtpApplyLinearBlockUpdatesKernel<<<grid_size, block_size, 0, stream>>>(
        kv_cache_kernel_block_id.data_ptr<int32_t>(),
        linear_group_ids.data_ptr<int32_t>(),
        linear_block_updates.data_ptr<int32_t>(),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(kv_cache_kernel_block_id.size(0)),
        static_cast<int32_t>(linear_group_num),
        static_cast<int32_t>(kv_cache_kernel_block_id.size(2)),
        kernel_blocks_per_kv_block);
}

}  // namespace rtp_llm
