#pragma once

#include <functional>
#include <memory>
#include <optional>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"

namespace rtp_llm {

class KVCacheManager;
struct GptModelInitParams;

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams&                params,
                            const std::shared_ptr<KVCacheManager>& cache_manager,
                            bool                                   warm_up                 = false,
                            bool                                   is_propose              = false,
                            int                                    propose_model_index     = 0,
                            MlaOpsType                             mla_ops_type            = MlaOpsType::AUTO,
                            int32_t                                kv_cache_group_num      = 1,
                            const std::vector<int32_t>&            kv_cache_layer_to_group = {});
    ~NormalExecutor();
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void         reportMetrics(const StreamGroups&             stream_groups,
                               RtpLLMExecutorMetricsCollector& executor_collector,
                               RtpLLMTokenPSMetricsCollector&  tps_collector);

    void setBatchProcessor(std::unique_ptr<NormalBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    // Test hook: if set, used to create model when py_model is None
    using ModelFactory = std::function<std::unique_ptr<ModelBase>(const GptModelInitParams&)>;
    static ModelFactory test_model_factory;

    bool updateEplbConfig(const EPLBConfig& config) override;

protected:
    // Stream-async dispatch gate. Reuses the common async env var so a
    // single launcher knob (`RTP_LLM_STREAM_ASYNC=1`) flips async scheduling.
    // Default off — production behaviour unchanged unless explicitly enabled.
    bool useStreamAsync() const;

    // Skip the front-loaded previous-worker sync when DROP_BROAD_SYNC=1.
    // Host stream state may still be mutating; NormalAsyncDeviceState carries
    // the sampled token and seq_len needed to launch the next normal decode
    // forward. Sampler still waits for committed host token history.
    bool useDropBroadSync() const;

    // Stream-async dispatch. Records sampler_event on the main stream after
    // sampler_->forward, then forks the bookkeeping worker to wait on the
    // event and run dispatch + per-stream update off the main thread.
    absl::Status dispatchOutputAsync(const StreamGroups&           stream_groups,
                                     GptModelOutputs               model_output,
                                     SamplerOutput                 sampler_output,
                                     std::shared_ptr<torch::Event> sampler_event);

    void publishNormalDeviceState(const StreamGroups& stream_groups, const SamplerOutput& sampler_output);
    void prepareGrpcNormalDeviceState(const StreamGroups& stream_groups);

    // Gate the forward-only snapshot path before building StreamGroups; when
    // false and bookkeeping is pending, process() waits before reading host state.
    bool canUseDeviceStateBeforeStreamGroups(const std::list<GenerateStreamPtr>& streams) const;

    void ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag);

private:
    std::unique_ptr<ModelBase>                                               model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<NormalBatchStreamProcessor>                              batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    RoleType                                                                 role_type_ = RoleType::PDFUSION;
    bool                                                                     warm_up_;
    bool                                                                     use_all_gather_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    bool                                                                     enable_ffn_disaggregate_ = false;
    bool                                                                     enable_detail_log_       = false;

    bool              is_propose_          = false;
    int               propose_model_index_ = 0;
    int               tp_rank_             = 0;
    ParallelismConfig parallelism_config_;

    // Stream-async worker owns a CUDA stream/thread for pinned D2H,
    // per-stream update, and KV release off the main thread.
    AsyncRunner dispatch_runner_;

    // Keeps async copy source tensors alive across release points. NormalExecutor
    // uses this for model-input H2D staging and sampler-input staging.
    TensorHolder buffer_holder_;
};

}  // namespace rtp_llm
