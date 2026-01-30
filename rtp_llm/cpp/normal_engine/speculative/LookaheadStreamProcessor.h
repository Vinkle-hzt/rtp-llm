#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

class LookaheadStreamProcessor: public NormalBatchStreamProcessor {
public:
    LookaheadStreamProcessor(const ModelConfig&                 model_config,
                             const PDSepConfig&                 pd_sep_config,
                             const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                             const CacheConfig&                 cache_config,
                             const SpeculativeExecutionConfig&  sp_config,
                             bool                               warm_up):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
        propose_step_(sp_config.gen_num_per_cycle) {}

    void updateSpecModelInput(const StreamGroups& stream_groups,
                              GptModelInputs&     model_input,
                              const size_t        total_propose_tokens) const;

    void updateDraftSamplerOutput(const StreamGroups& stream_groups,
                                  SamplerOutput&      draft_sampler_output,
                                  torch::Tensor&      draft_token_ids_d_t,
                                  torch::Tensor&      draft_token_probs_d_t) const;

    absl::StatusOr<SamplerInputs> gatherSpecSamplerInput(const StreamGroups&    stream_groups,
                                                         const GptModelInputs&  model_inputs,
                                                         const GptModelOutputs& model_output,
                                                         const size_t           total_propose_tokens) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    void prepareDecodeSpecUpdateInfo(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     const MergedOutput&                          draft_prefill_output,
                                     std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const;

protected:
    int propose_step_;
};
}  // namespace rtp_llm
