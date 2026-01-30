#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeUtil.h"

namespace rtp_llm {

class LookaheadProposer {
public:
    LookaheadProposer(size_t min_token_match_len, size_t max_token_match_len, size_t propose_step, DeviceBase* device):
        min_token_match_len_(min_token_match_len),
        max_token_match_len_(max_token_match_len),
        propose_step_(propose_step),
        device_(device) {}

    ~LookaheadProposer() {};

    void forward(const std::list<GenerateStreamPtr>& streams,
                 GptModelInputs&                     model_input,
                 SpecMetricsCollectors&              metrics_collector);

    void ruleBasedTokenSelector(const GenerateStreamPtr& stream);

    void SpEditTokenSelector(const GenerateStreamPtr&            stream,
                             SpeculativeExecutorStreamOutputPtr& stream_output,
                             bool                                use_sp_advice_prompt);

    void PromptLookUpTokenSelector(const GenerateStreamPtr&            stream,
                                   SpeculativeExecutorStreamOutputPtr& stream_output,
                                   bool                                use_sp_advice_prompt);

    void postProcess(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

private:
    size_t min_token_match_len_;
    size_t max_token_match_len_;
    size_t propose_step_;

    DeviceBase* device_;

    // holder for host buffers to avoid early free before H2D copy kernel execution
    ModelBufferHolder buffer_holder_;
};

}  // namespace rtp_llm