#include "rtp_llm/cpp/normal_engine/speculative/LookaheadStreamProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <cstring>

namespace rtp_llm {

void LookaheadStreamProcessor::updateSpecModelInput(const StreamGroups& stream_groups,
                                                    GptModelInputs&     model_input,
                                                    const size_t        total_propose_tokens) const {
    size_t    batch_size = stream_groups.size();
    BufferPtr combo_tokens =
        device_->allocateBuffer({DataType::TYPE_INT32, {total_propose_tokens + batch_size}, AllocationType::HOST});
    BufferPtr prefix_lengths = device_->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST});
    BufferPtr lm_output_indexes =
        device_->allocateBuffer({DataType::TYPE_INT32, {total_propose_tokens + batch_size}, AllocationType::HOST});

    int* combo_token_ids = combo_tokens->data<int>();

    int batch_idx       = 0;
    int combo_token_idx = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer                  = stream->getSPOutputBuffer();
        int  cur_propose_step                  = sp_output_buffer->propose_step;
        prefix_lengths->data<int>()[batch_idx] = model_input.input_lengths->data<int>()[batch_idx] =
            cur_propose_step + 1;

        combo_token_ids[combo_token_idx] = model_input.combo_tokens->data<int>()[batch_idx];
        for (int i = 0; i < cur_propose_step; i++) {
            combo_token_ids[combo_token_idx + i + 1] = sp_output_buffer->tokens->data<int>()[i];
        }

        combo_token_idx += cur_propose_step + 1;
        batch_idx += 1;
    }

    for (int i = 0; i < total_propose_tokens + batch_size; i++) {
        lm_output_indexes->data<int>()[i] = i;
    }

    model_input.combo_tokens      = combo_tokens;
    model_input.prefix_lengths    = model_input.sequence_lengths;
    model_input.sequence_lengths  = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
    model_input.lm_output_indexes = lm_output_indexes;
}

absl::StatusOr<SamplerInputs>
LookaheadStreamProcessor::gatherSpecSamplerInput(const StreamGroups&    stream_groups,
                                                 const GptModelInputs&  model_inputs,
                                                 const GptModelOutputs& model_output,
                                                 const size_t           total_propose_tokens) const {
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams      = stream_groups.allStreams();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t total_batch_size = stream_groups.size() + total_propose_tokens;

    SamplerInputs sampler_inputs = allocateSamplerInputs(
        stream_groups, total_batch_size, total_batch_size, model_inputs.sequence_lengths, propose_step_);
    setCommonSamplerInputs(sampler_inputs, all_streams, true, propose_step_);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        auto        sp_output_buffer   = stream->getSPOutputBuffer();
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = sp_output_buffer->propose_step + 1;

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(0),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }

        RTP_LLM_LOG_DEBUG("stream [%ld], complete token ids = [%s]",
                          stream->streamId(),
                          complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size       = model_output.logits->shape()[1];
    sampler_inputs.logits = device_->allocateBuffer(
        {model_output.logits->type(), {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
    if (return_all_probs) {
        sampler_inputs.all_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
    }

    device_->copy({*sampler_inputs.logits, *model_output.logits});

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      device_->clone({*sampler_inputs.logits, rtp_llm::AllocationType::HOST})
                          ->debugStringWithData<float>(10)
                          .c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

void LookaheadStreamProcessor::updateDraftSamplerOutput(const StreamGroups& stream_groups,
                                                        SamplerOutput&      draft_sampler_output,
                                                        torch::Tensor&      draft_token_ids_d_t,
                                                        torch::Tensor&      draft_token_probs_d_t) const {

    std::vector<torch::Tensor> draft_token_probs_list;
    std::vector<torch::Tensor> draft_token_ids_list;

    for (auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        int  cur_propose_step = sp_output_buffer->propose_step;
        if (cur_propose_step > 0) {
            auto token_ids   = Buffer2torchTensor(sp_output_buffer->tokens, false);
            auto token_probs = Buffer2torchTensor(sp_output_buffer->all_probs, false);
            draft_token_ids_list.push_back(token_ids);
            draft_token_probs_list.push_back(token_probs);
        }
    }

    if (!draft_token_ids_list.empty()) {
        draft_token_ids_d_t   = torch::cat(draft_token_ids_list, 1).contiguous();
        draft_token_probs_d_t = torch::cat(draft_token_probs_list, 0).reshape({-1, (int)vocab_size_}).contiguous();

        draft_sampler_output.token_ids = torchTensor2Buffer(draft_token_ids_d_t);
        draft_sampler_output.all_probs = torchTensor2Buffer(draft_token_probs_d_t);
    } else {
        draft_sampler_output.token_ids = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
        draft_sampler_output.all_probs =
            device_->allocateBuffer({DataType::TYPE_FP32, {0, vocab_size_}, AllocationType::HOST});
    }
}

absl::Status LookaheadStreamProcessor::dispatchDecode(const StreamGroups&                          stream_groups,
                                                      const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                                      const MergedOutput& draft_prefill_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    prepareDecodeSpecUpdateInfo(stream_groups, spec_decode_output, draft_prefill_output, spec_update_infos);

    stream_groups.updateStreams(spec_update_infos);

    return absl::OkStatus();
}

void LookaheadStreamProcessor::prepareDecodeSpecUpdateInfo(
    const StreamGroups&                          stream_groups,
    const speculative::SpeculativeSamplerOutput& spec_decode_output,
    const MergedOutput&                          draft_prefill_output,
    std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const {
    const auto& accept_len    = spec_decode_output.accept_len;
    const auto& accept_tokens = spec_decode_output.accept_tokens;

    const auto& draft_model_output   = draft_prefill_output.model_output;
    const auto& draft_sampler_output = draft_prefill_output.sampler_output;

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        // lookahead not need to transfer hidden states and token probs
        BufferPtr propose_all_probs  = nullptr;
        BufferPtr last_hidden_states = nullptr;

        spec_update_infos.push_back({accept_tokens[batch_idx_out],
                                     accept_len[batch_idx_out],
                                     -1,
                                     std::move(last_hidden_states),
                                     std::move(propose_all_probs)});

        token_offset += accept_len[batch_idx_out];
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}

}  // namespace rtp_llm
