#include "rtp_llm/cpp/normal_engine/speculative/LookaheadExecutor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {
LookaheadExecutor::LookaheadExecutor(const EngineInitParams&                        params,
                                     std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                                     const std::shared_ptr<KVCacheManager>&         cache_manager,
                                     rtp_llm::DeviceBase*                           device,
                                     const std::shared_ptr<lora::LoraManager>&      lora_manager,
                                     bool                                           warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    metrics_reporter_(params.metrics_reporter),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type) {
    propose_step_        = propose_params->gen_num_per_circle;
    vocab_size_          = params.model_config_.vocab_size;
    min_token_match_len_ = params.sp_config.sp_min_token_match;
    max_token_match_len_ = params.sp_config.sp_max_token_match;

    RTP_LLM_LOG_INFO(
        "[speculative decoding] using lookahead executor, min_token_match_len_ = %d, max_token_match_len_ = %d, propose_step_ = %d",
        min_token_match_len_,
        max_token_match_len_,
        propose_step_);

    lookahead_proposer_.reset(
        new LookaheadProposer(min_token_match_len_, max_token_match_len_, propose_step_, device_));

    enable_detail_log_ = params.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = params.model_config_.moe_layer_index.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();
        bool is_gated_activation = params.model_config_.isGatedActivation();
        auto moe_inter_size =
            is_gated_activation ?
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1] / 2 :
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1];

        expert_balancer_ =
            std::make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                             params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                             params.model_config_.num_layers,
                                             moe_inter_size,
                                             params.model_config_.hidden_size,
                                             params.parallelism_config.ep_rank,
                                             params.parallelism_config.ep_size,
                                             params.py_eplb,
                                             moe_weight_type,
                                             device_,
                                             params.model_config_.quant_algo,
                                             metrics_reporter_,
                                             params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{device_}));
    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(cache_manager->kvCacheBuffer()) : std::nullopt,
         params.model_id});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
        model_.reset(new NativeDeviceGraphModel(model_init_params));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new LookaheadStreamProcessor(params.model_config_,
                                                               params.pd_sep_config,
                                                               params.profiling_debug_logging_config,
                                                               cache_config,
                                                               params.sp_config,
                                                               warm_up_));

    LogitsProcessorFactory::init(params.model_config_.ckpt_path, params.sp_config.tree_decode_config);
    speculative_sampler_.reset(new speculative::SpeculativeSampler(device, nullptr, propose_step_));

    device_->profileStart();
}

bool LookaheadExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return false;
}

bool LookaheadExecutor::isTpRank0() const {
    return device_->getDeviceProperties().tp_rank == 0;
}

void LookaheadExecutor::maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const {
    bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
    if (force) {
        RTP_LLM_LOG_INFO("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    } else {
        RTP_LLM_LOG_DEBUG("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    }
}

absl::Status LookaheadExecutor::prefillStep(const std::list<GenerateStreamPtr>& streams,
                                            SpecMetricsCollectors&              metrics_collector) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;

    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;
    // make sure last model input is released before forward
    model_->releaseBuffers();

    // update kv cache
    if (model_input.kv_cache_update_mapping) {
        cache_manager_->blockBatchCopy(*model_input.kv_cache_update_mapping);
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input =
            lora_manager_->makeLoraModelInput(model_input.lora_ids, model_input.lora_input_lengths);
    }

    {
        maybePrintModelInput(model_input, "prefill target model");
        model_output = std::move(model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[speculative][lookahead][prefill] model forward done");
    }

    // eplb
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_ || streams.size() == 0) {
        device_->syncAndCheck();
        model_->releaseBuffers();
        return absl::OkStatus();
    }

    {
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("[speculative][lookahead][prefill] sampler forward done");
    }

    {
        auto result =
            batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
        RTP_LLM_LOG_DEBUG("[speculative][lookahead][prefill] dispatch done");

        model_->releaseBuffers();

        return result;
    }
}

absl::Status LookaheadExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                           SpecMetricsCollectors&              metrics_collector) {
    RtpLLMExecutorMetricsCollector&          executor_collector  = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&           tps_collector       = metrics_collector.tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector& sp_engine_collector = metrics_collector.sp_engine_collector;

    StreamGroups                          stream_groups(streams);
    GptModelInputs                        model_input;
    GptModelOutputs                       model_output;
    SamplerOutput                         sampler_output;
    SamplerOutput                         draft_sampler_output;
    BufferPtr                             cu_num_spec_tokens_host;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    size_t total_propose_tokens = 0;

    torch::Tensor draft_token_ids_d_t;
    torch::Tensor draft_token_probs_d_t;

    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        model_input.skip_run                     = streams.empty() && !enable_ffn_disaggregate_;
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // use lookahead to generate tokens
    if (!model_input.skip_run && !model_input.is_fake_stream && device_->getDeviceProperties().tp_rank == 0) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        lookahead_proposer_->forward(streams, model_input, metrics_collector);
        total_propose_tokens = stream_groups.totalProposeTokens();
        batch_stream_processor_->updateSpecModelInput(stream_groups, model_input, total_propose_tokens);
        sp_engine_collector.propose_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // make sure last model input is released before forward
    model_->releaseBuffers();

    // update kv cache
    if (model_input.kv_cache_update_mapping) {
        cache_manager_->blockBatchCopy(*model_input.kv_cache_update_mapping);
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input =
            lora_manager_->makeLoraModelInput(model_input.lora_ids, model_input.lora_input_lengths);
    }

    {
        maybePrintModelInput(model_input, "target model spec decode");
        model_output = std::move(model_->forward(model_input));
        RTP_LLM_LOG_DEBUG("[speculative][lookahead][decode] model forward done");
    }

    // prepare spec sampler input
    if (device_->getDeviceProperties().tp_rank == 0) {
        batch_stream_processor_->updateDraftSamplerOutput(
            stream_groups, draft_sampler_output, draft_token_ids_d_t, draft_token_probs_d_t);
    }

    // eplb
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_ || streams.size() == 0) {
        device_->syncAndCheck();
        model_->releaseBuffers();
        return absl::OkStatus();
    }

    // prepare draft sample output
    {
        if (model_input.is_fake_stream) {
            BufferPtr accept_tokens = device_->allocateBuffer({DataType::TYPE_INT32, {1, 1}, AllocationType::HOST});
            *accept_tokens->dataWithOffset<int32_t>(0) = 0;
            speculative_sampler_output.accept_len      = {1};
            speculative_sampler_output.accept_tokens   = {std::move(accept_tokens)};
            device_->syncAndCheck();
        } else {
            CHECK_AND_RETURN_REF(sampler_input,
                                 batch_stream_processor_->gatherSpecSamplerInput(
                                     stream_groups, model_input, model_output, total_propose_tokens));
            sampler_output = std::move(sampler_->forward(sampler_input));
            RTP_LLM_LOG_DEBUG("[speculative][lookahead][decode] sampler forward done");
            size_t batch_size = stream_groups.size();
            cu_num_spec_tokens_host =
                device_->allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}, AllocationType::HOST});

            int cu_num_spec_tokens                  = 0;
            int stream_idx                          = 0;
            cu_num_spec_tokens_host->data<int>()[0] = 0;
            for (auto& stream : stream_groups.allStreams()) {
                cu_num_spec_tokens += stream->getSPOutputBuffer()->propose_step;
                cu_num_spec_tokens_host->data<int>()[stream_idx + 1] = cu_num_spec_tokens;
                stream_idx++;
            }

            // rejection sampling
            speculative_sampler_output =
                speculative_sampler_->forward(streams, draft_sampler_output, sampler_output, cu_num_spec_tokens_host);
            RTP_LLM_LOG_DEBUG("[speculative][lookahead][decode] speculative sampler forward done");
        }
    }

    {
        auto result = batch_stream_processor_->dispatchDecode(
            stream_groups, speculative_sampler_output, {std::move(model_output), std::move(sampler_output)});
        RTP_LLM_LOG_DEBUG("[speculative][lookahead][decode] dispatch done");

        model_->releaseBuffers();

        return result;
    }
}

absl::Status LookaheadExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    SpecMetricsCollectors metrics_collector;

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    // prepare streams
    prepareStreams(streams, prefill_streams, decode_streams);

    // step forward
    int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(prefillStep(prefill_streams, metrics_collector));
    }

    if (role_type_ == RoleType::DECODE || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(decodeStep(decode_streams, metrics_collector));
    }

    metrics_collector.sp_engine_collector.step_latency_us =
        autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;

    // report metrics
    if (isTpRank0() && metrics_reporter_ && metrics_collector.not_skip) {
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(
            nullptr, &metrics_collector.executor_collector);
        metrics_reporter_->report<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
            nullptr, &metrics_collector.tps_collector);
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(
            nullptr, &metrics_collector.sp_engine_collector);
    }

    return absl::OkStatus();
}

void LookaheadExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams,
                                       std::list<GenerateStreamPtr>&       prefill_streams,
                                       std::list<GenerateStreamPtr>&       decode_streams) {
    for (auto& stream : streams) {
        if (stream->isContextStream()) {
            prefill_streams.push_back(stream);
        } else {
            if (!stream->getSPOutputBuffer()) {
                SpeculativeExecutorStreamOutputPtr sp_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
                stream->setSPOutputBuffer(sp_buffer);
            }
            decode_streams.push_back(stream);
        }
        stream->setReturnAllProbs(true);
    }
}
}  // namespace rtp_llm
