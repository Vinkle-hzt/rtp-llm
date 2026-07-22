#include <gtest/gtest.h>
#include <torch/extension.h>

#include <set>
#include <vector>

#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"
#include "rtp_llm/models_py/bindings/common/kernels/mtp_target_verify_prepare.h"

namespace rtp_llm {
namespace {

TEST(MtpLinearBlockUpdatesTest, MatchesOrderedHostSwapsForMultipleGroups) {
    constexpr int32_t          group_count                = 3;
    constexpr int32_t          batch_size                 = 4;
    constexpr int32_t          physical_block_count       = 8;
    constexpr int32_t          seq_size_per_block         = 4;
    constexpr int32_t          kernel_blocks_per_kv_block = 2;
    constexpr int32_t          kernel_block_count         = physical_block_count * kernel_blocks_per_kv_block;
    const std::vector<int32_t> linear_group_ids           = {0, 2};
    const std::vector<int32_t> accept_len                 = {1, 2, 4, 5};
    const std::vector<int32_t> prev_seq_len               = {5, 4, 3, 8};

    std::vector<int32_t> physical(group_count * batch_size * physical_block_count);
    auto                 physical_offset = [](int32_t group, int32_t batch, int32_t pos) {
        return (group * batch_size + batch) * physical_block_count + pos;
    };
    for (int32_t group = 0; group < group_count; ++group) {
        for (int32_t batch = 0; batch < batch_size; ++batch) {
            for (int32_t pos = 0; pos < physical_block_count; ++pos) {
                physical[physical_offset(group, batch, pos)] = 100 * group + 10 * batch + pos + 1;
            }
        }
    }
    physical[physical_offset(2, 2, 0)] = -1;

    auto expected = physical;
    for (int32_t batch = 0; batch < batch_size; ++batch) {
        if (accept_len[batch] <= 1) {
            continue;
        }
        const int32_t cur_cached_len = prev_seq_len[batch] - 1;
        const int32_t nxt_cached_len = cur_cached_len + accept_len[batch];
        const auto [cached_src, cached_dst] =
            getCachedTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, seq_size_per_block);
        const auto [final_src, final_dst] =
            getFinalTokenBlockSwapIdx(cur_cached_len, nxt_cached_len, seq_size_per_block);
        for (int32_t group : linear_group_ids) {
            std::swap(expected[physical_offset(group, batch, cached_src)],
                      expected[physical_offset(group, batch, cached_dst)]);
            std::swap(expected[physical_offset(group, batch, final_src)],
                      expected[physical_offset(group, batch, final_dst)]);
        }
    }

    auto expand_kernel_blocks = [&](const std::vector<int32_t>& physical_ids) {
        std::vector<int32_t> kernel_ids(group_count * batch_size * kernel_block_count);
        for (int32_t group = 0; group < group_count; ++group) {
            for (int32_t batch = 0; batch < batch_size; ++batch) {
                for (int32_t pos = 0; pos < physical_block_count; ++pos) {
                    const int32_t physical_id = physical_ids[physical_offset(group, batch, pos)];
                    for (int32_t slot = 0; slot < kernel_blocks_per_kv_block; ++slot) {
                        const int32_t kernel_offset =
                            (group * batch_size + batch) * kernel_block_count + pos * kernel_blocks_per_kv_block + slot;
                        kernel_ids[kernel_offset] =
                            physical_id < 0 ? -1 : physical_id * kernel_blocks_per_kv_block + slot;
                    }
                }
            }
        }
        return kernel_ids;
    };

    const auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       block_table = torch::tensor(expand_kernel_blocks(physical), torch::kInt32)
                           .reshape({group_count, batch_size, kernel_block_count})
                           .to(torch::kCUDA);
    auto accept_len_gpu       = torch::tensor(accept_len, cuda_i32);
    auto prev_seq_len_gpu     = torch::tensor(prev_seq_len, cuda_i32);
    auto linear_group_ids_gpu = torch::tensor(linear_group_ids, cuda_i32);
    auto updates = torch::empty({batch_size, static_cast<int64_t>(linear_group_ids.size()), 4, 2}, cuda_i32);

    invokeMtpBuildLinearBlockUpdates(accept_len_gpu,
                                     prev_seq_len_gpu,
                                     block_table,
                                     linear_group_ids_gpu,
                                     updates,
                                     seq_size_per_block,
                                     kernel_blocks_per_kv_block,
                                     MtpPrepareStream{});
    invokeMtpApplyLinearBlockUpdates(
        block_table, linear_group_ids_gpu, updates, kernel_blocks_per_kv_block, MtpPrepareStream{});

    auto expected_tensor = torch::tensor(expand_kernel_blocks(expected), torch::kInt32)
                               .reshape({group_count, batch_size, kernel_block_count});
    EXPECT_TRUE(torch::equal(block_table.cpu(), expected_tensor));

    auto once_applied = block_table.clone();
    invokeMtpApplyLinearBlockUpdates(
        block_table, linear_group_ids_gpu, updates, kernel_blocks_per_kv_block, MtpPrepareStream{});
    EXPECT_TRUE(torch::equal(block_table.cpu(), once_applied.cpu()));

    auto updates_cpu = updates.cpu();
    auto accessor    = updates_cpu.accessor<int32_t, 4>();
    for (int32_t batch = 0; batch < batch_size; ++batch) {
        for (int32_t group = 0; group < static_cast<int32_t>(linear_group_ids.size()); ++group) {
            std::set<int32_t> positions;
            for (int32_t update = 0; update < 4; ++update) {
                const int32_t pos = accessor[batch][group][update][0];
                if (pos >= 0) {
                    EXPECT_TRUE(positions.insert(pos).second);
                }
            }
        }
    }
}

}  // namespace
}  // namespace rtp_llm
