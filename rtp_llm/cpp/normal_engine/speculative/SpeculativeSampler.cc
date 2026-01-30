#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {
namespace speculative {

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;
    output.all_probs = torch::softmax(logits, -1);

    std::tuple<torch::Tensor, torch::Tensor> sample_res;
    if (top_k == 1) {
        sample_res = torch::max(output.all_probs, -1, true);
    } else {
        sample_res = torch::topk(output.all_probs, top_k, -1);
    }

    output.token_ids = std::get<1>(sample_res);

    int batch_size = output.token_ids.size(0);
    device_->mappingDraft2Target({torchTensor2Buffer(output.token_ids), d2t_map_, batch_size, 0, 1});

    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output,
                                                     BufferPtr                           cu_num_spec_tokens_host) {
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output, cu_num_spec_tokens_host);

    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output,
                                     BufferPtr                           cu_num_spec_tokens_host) const {
    torch::Device target_device = device_->getTorchDevice();
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    const int* new_all_token_ids  = target_sampler_output.token_ids->data<int32_t>();
    const int* cu_num_spec_tokens = cu_num_spec_tokens_host->data<int32_t>();

    const size_t token_stride         = target_sampler_output.token_ids->shape()[1];
    const int    total_spec_token_num = cu_num_spec_tokens[batch_size];

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    BufferPtr draft_token_ids_d;
    BufferPtr target_token_ids_d;
    BufferPtr cu_num_spec_tokens_d = device_->clone({*cu_num_spec_tokens_host, AllocationType::DEVICE});

    if (draft_token_ids->where() != MemoryType::MEMORY_GPU) {
        draft_token_ids_d = device_->clone({*draft_token_ids, AllocationType::DEVICE});
    } else {
        draft_token_ids_d = draft_token_ids;
    }

    if (target_token_ids->where() != MemoryType::MEMORY_GPU) {
        target_token_ids_d = device_->clone({*target_token_ids, AllocationType::DEVICE});
    } else {
        target_token_ids_d = target_token_ids;
    }

    torch::Tensor do_sample  = torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool));
    int           stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample[stream_idx] = !stream->generateConfig()->top1();
        stream_idx++;
    }
    auto do_sample_d = do_sample.to(target_device, true, false);

    // note target token probs is already on device
    auto target_token_probs_d = target_token_probs;
    auto draft_token_probs_d  = draft_token_probs;

    // prepare data for chain speculative sampling
    auto draft_token_ids_d_t    = Buffer2torchTensor(draft_token_ids_d, false);
    auto draft_token_probs_d_t  = Buffer2torchTensor(draft_token_probs_d, false);
    auto target_token_ids_d_t   = Buffer2torchTensor(target_token_ids_d, false);
    auto target_token_probs_d_t = Buffer2torchTensor(target_token_probs_d, false);
    auto cu_num_spec_tokens_d_t = Buffer2torchTensor(cu_num_spec_tokens_d, false);

    torch::Tensor uniform_samples_d =
        torch::rand({total_spec_token_num + batch_size},
                    torch::TensorOptions().device(target_device).dtype(torch::kFloat).requires_grad(false));
    torch::Tensor output_token_ids_d =
        torch::zeros({total_spec_token_num + batch_size},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    if (draft_token_probs_d_t.size(-1) != target_token_probs_d_t.size(-1)) {
        std::cout << "draft_token_probs_d_t.size(-1) != target_token_probs_d_t.size(-1)"
                  << draft_token_probs_d_t.size(-1) << " != " << target_token_probs_d_t.size(-1) << std::endl;
        auto draft_probs_padding =
            torch::zeros({(long)batch_size, draft_token_probs_d_t.size(1), target_token_probs_d_t.size(2)},
                         draft_token_probs_d_t.options());
        torch::Tensor d2t_map_d_t = Buffer2torchTensor(d2t_map_, false);
        // draft_probs_padding[:, :, d2t_map_d_t] = draft_probs_d_t
        draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_d_t},
                                       draft_token_probs_d_t);
        draft_token_probs_d_t = draft_probs_padding;
    }

    device_->rejectionSampling({
        draft_token_probs_d_t,
        draft_token_ids_d_t,
        uniform_samples_d,
        target_token_probs_d_t,
        target_token_ids_d_t,
        output_token_ids_d,
        output_accepted_token_num_d,
        do_sample_d,
        cu_num_spec_tokens_d_t,
    });

    // back to host
    torch::Tensor output_token_ids_h          = output_token_ids_d.to(host_device).contiguous();
    torch::Tensor output_accepted_token_num_h = output_accepted_token_num_d.to(host_device).contiguous();

    BufferPtr draft_token_ids_h;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->forceSpAccept()) {
            draft_token_ids_h = device_->clone({*draft_token_ids, AllocationType::HOST});
            break;
        }
    }

    stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        BufferPtr accept_tokens;
        size_t    accept_len           = 0;
        size_t    current_propose_step = cu_num_spec_tokens[stream_idx + 1] - cu_num_spec_tokens[stream_idx];

        if (stream->forceSpAccept()) {
            accept_len    = current_propose_step + 1;
            accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});
            memcpy(accept_tokens->data(),
                   draft_token_ids_h->dataWithOffset<int32_t>(cu_num_spec_tokens[stream_idx]),
                   sizeof(int32_t) * current_propose_step);
            *accept_tokens->dataWithOffset<int32_t>(accept_len - 1) =
                new_all_token_ids[(cu_num_spec_tokens[stream_idx] + stream_idx + accept_len - 1) * token_stride
                                  + token_stride - 1];
        } else {
            accept_len    = output_accepted_token_num_h[stream_idx].item<int32_t>();
            accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});
            memcpy(accept_tokens->data(),
                   output_token_ids_h.data_ptr<int32_t>() + cu_num_spec_tokens[stream_idx] + stream_idx,
                   sizeof(int32_t) * accept_len);
        }

        sample_output.accept_tokens.push_back(accept_tokens);
        sample_output.accept_len.push_back(accept_len);
        stream_idx++;
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}
}  // namespace speculative
}  // namespace rtp_llm