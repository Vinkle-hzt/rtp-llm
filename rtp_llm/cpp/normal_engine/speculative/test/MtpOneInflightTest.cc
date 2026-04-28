#include <cstdlib>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#undef protected
#undef private
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

struct OneInflightFixture {
    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
};

std::unique_ptr<ProposeModelEngineInitParams> makeProposeParams(const EngineInitParams& params,
                                                                size_t                  gen_num_per_cycle) {
    auto mtp_model_params   = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    auto mtp_params         = std::make_unique<EngineInitParams>(params);
    mtp_params->py_sp_model = py::none();
    mtp_model_params->push_back(std::move(mtp_params));
    return std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, gen_num_per_cycle, std::move(mtp_model_params));
}

std::unique_ptr<MtpExecutor> makeExecutor(size_t gen_num_per_cycle, OneInflightFixture* fixture) {
    CustomConfig               config;
    KVCacheConfig              kv_cache_config;
    SpeculativeExecutionConfig sp_config;

    fixture->model_config.max_seq_len = 64;
    fixture->model_config.vocab_size  = 8;
    fixture->model_config.num_layers  = 1;
    sp_config.gen_num_per_cycle       = gen_num_per_cycle;

    auto cache_config     = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                       /*block_num=*/16,
                                                       /*tokens_per_block=*/2,
                                                       rtp_llm::TYPE_INT8,
                                                       /*local_head_num_kv=*/128,
                                                       /*size_per_head=*/256);
    auto mtp_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                           /*block_num=*/16,
                                                           /*tokens_per_block=*/2,
                                                           rtp_llm::TYPE_INT8,
                                                           /*local_head_num_kv=*/128,
                                                           /*size_per_head=*/256);
    cache_config.mtp_sub_configs.push_back(std::make_shared<CacheConfig>(mtp_cache_config));

    EngineInitParams params =
        createEngineInitParams(config, fixture->model_config, fixture->runtime_config, kv_cache_config);
    params.sp_config                  = sp_config;
    params.pd_sep_config.role_type    = RoleType::PDFUSION;
    params.parallelism_config.tp_size = 1;
    params.parallelism_config.tp_rank = 0;
    params.parallelism_config.dp_size = 1;
    params.parallelism_config.pp_size = 1;
    params.py_model                   = py::none();
    params.py_sp_model                = py::none();

    auto propose_params = makeProposeParams(params, gen_num_per_cycle);
    auto cache_manager  = std::make_shared<KVCacheManager>(cache_config);
    EXPECT_TRUE(cache_manager->init());
    fixture->resource_context.cache_manager = cache_manager;
    fixture->resource_context.role_type     = RoleType::PDFUSION;

    return std::make_unique<MtpExecutor>(params, propose_params, cache_manager);
}

GenerateStreamPtr makeDecodeStream(const OneInflightFixture& fixture) {
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 16;

    auto stream = std::make_shared<NormalGenerateStream>(
        input, fixture.model_config, fixture.runtime_config, fixture.resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    *stream->is_context_stream_      = false;
    stream->setNeedReleaseResource(false);

    return stream;
}

uint64_t attachCudaOneInflightState(const GenerateStreamPtr& stream) {
    return stream->setSpecDecodeDeviceState(torch::tensor({1}, torch::kInt32).to(torch::kCUDA),
                                            torch::tensor({{2, 3}}, torch::kInt32).to(torch::kCUDA),
                                            torch::tensor({4}, torch::kInt32).to(torch::kCUDA),
                                            torch::tensor({{5}}, torch::kInt32).to(torch::kCUDA));
}

TEST(MtpOneInflightTest, CanSkipPrevBookkeepingWhenAllOneInflightGuardsPass) {
    setenv("RTP_LLM_MTP_STREAM_ASYNC", "1", /*overwrite=*/1);
    setenv("RTP_LLM_ASYNC_PROCESS_BEFORE_AWAIT", "1", /*overwrite=*/1);

    OneInflightFixture fixture;
    auto               executor = makeExecutor(/*gen_num_per_cycle=*/1, &fixture);
    auto               stream   = makeDecodeStream(fixture);
    auto               epoch    = attachCudaOneInflightState(stream);

    EXPECT_TRUE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));

    EXPECT_TRUE(stream->clearSpecDecodeDeviceState(epoch));
}

TEST(MtpOneInflightTest, RejectsStreamsWithoutDeviceStateOrWithPendingSwap) {
    setenv("RTP_LLM_MTP_STREAM_ASYNC", "1", /*overwrite=*/1);
    setenv("RTP_LLM_ASYNC_PROCESS_BEFORE_AWAIT", "1", /*overwrite=*/1);

    OneInflightFixture fixture;
    auto               executor = makeExecutor(/*gen_num_per_cycle=*/1, &fixture);
    auto               stream   = makeDecodeStream(fixture);

    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));

    auto epoch = attachCudaOneInflightState(stream);
    stream->setPendingSwapDoneEvent(std::make_shared<int>(1));
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));

    stream->clearPendingSwapDoneEvent();
    EXPECT_TRUE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));

    EXPECT_TRUE(stream->clearSpecDecodeDeviceState(epoch));
}

TEST(MtpOneInflightTest, RejectsContextInactiveAndUnsupportedExecutorShapes) {
    setenv("RTP_LLM_MTP_STREAM_ASYNC", "1", /*overwrite=*/1);
    setenv("RTP_LLM_ASYNC_PROCESS_BEFORE_AWAIT", "1", /*overwrite=*/1);

    OneInflightFixture fixture;
    auto               executor = makeExecutor(/*gen_num_per_cycle=*/1, &fixture);
    auto               stream   = makeDecodeStream(fixture);
    auto               epoch    = attachCudaOneInflightState(stream);

    *stream->is_context_stream_ = true;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));
    *stream->is_context_stream_ = false;

    stream->generate_status_->status = StreamState::FINISHED;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));
    stream->generate_status_->status = StreamState::RUNNING;

    executor->propose_step_ = 2;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));
    executor->propose_step_ = 1;

    executor->parallelism_config_.tp_size = 2;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));
    executor->parallelism_config_.tp_size = 1;

    executor->role_type_ = RoleType::PREFILL;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));
    executor->role_type_ = RoleType::PDFUSION;

    executor->enable_ffn_disaggregate_ = true;
    EXPECT_FALSE(executor->canSkipPrevBookkeepingWaitForOneInflight({stream}));

    EXPECT_TRUE(stream->clearSpecDecodeDeviceState(epoch));
}

}  // namespace
}  // namespace rtp_llm
