#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeUtil.h"
#include "rtp_llm/cpp/normal_engine/speculative/LookaheadProposer.h"
#include "rtp_llm/cpp/normal_engine/speculative/LookaheadStreamProcessor.h"

namespace rtp_llm {
class LookaheadExecutor: public Executor {
public:
    explicit LookaheadExecutor(const EngineInitParams&                        params,
                               std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                               const std::shared_ptr<KVCacheManager>&         cache_manager,
                               rtp_llm::DeviceBase*                           device,
                               const std::shared_ptr<lora::LoraManager>&      lora_manager,
                               bool                                           warm_up = false);

    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    bool         updateEplbConfig(const EPLBConfig& config) override;

    void setTargetModel(std::unique_ptr<GptModel> model) {
        model_ = std::move(model);
    }

    void setBatchProcessor(std::unique_ptr<LookaheadStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setSpeculativeSampler(std::unique_ptr<speculative::SpeculativeSampler> sampler) {
        speculative_sampler_ = std::move(sampler);
    }

    void setSampler(std::unique_ptr<Sampler> sampler) {
        sampler_ = std::move(sampler);
    }

protected:
    bool isTpRank0() const;

    void maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const;

    absl::Status prefillStep(const std::list<GenerateStreamPtr>& streams, SpecMetricsCollectors& metrics_collector);

    absl::Status decodeStep(const std::list<GenerateStreamPtr>& streams, SpecMetricsCollectors& metrics_collector);

    void prepareStreams(const std::list<GenerateStreamPtr>& streams,
                        std::list<GenerateStreamPtr>&       prefill_streams,
                        std::list<GenerateStreamPtr>&       decode_streams);

private:
    std::unique_ptr<GptModel>                 model_;
    std::unique_ptr<Sampler>                  sampler_;
    std::unique_ptr<LookaheadStreamProcessor> batch_stream_processor_;
    std::shared_ptr<KVCacheManager>           cache_manager_;
    std::shared_ptr<lora::LoraManager>        lora_manager_;
    bool                                      enable_ffn_disaggregate_ = false;
    bool                                      enable_detail_log_       = false;
    kmonitor::MetricsReporterPtr              metrics_reporter_        = nullptr;
    std::shared_ptr<ExpertBalancer>           expert_balancer_;
    size_t                                    vocab_size_;

    // for mtp
    size_t                                           propose_step_;
    size_t                                           min_token_match_len_;
    size_t                                           max_token_match_len_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<LookaheadProposer>               lookahead_proposer_;

    bool     warm_up_;
    RoleType role_type_;
};
};  // namespace rtp_llm
