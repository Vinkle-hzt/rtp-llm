#pragma once

#include <atomic>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <torch/torch.h>

namespace rtp_llm {

enum class BatchFutureStage : int {
    Created = 0,
    Enqueued,
    Running,
    Done,
    Failed,
    TimedOut,
};

// Per-batch async handoff between the launch thread and the result thread.
//
// The synchronous NormalEngine::step() path issues schedule -> process ->
// result-bookkeeping back-to-back on the main loop. The async path lets an
// executor return a future that owns any stream/tensor/event state crossing
// that step boundary; the result thread consumes it and marks bookkeeping_done.
struct BatchFuture {
    virtual ~BatchFuture() = default;

    std::string debug_label = "normal_engine_batch";

    // Streams that participated in this batch. Owned references are kept
    // alive until bookkeeping completes so per-stream specUpdate / KV release
    // does not race with main-thread enqueue.
    std::list<GenerateStreamPtr> streams;

    // Timestamps are wall-clock micros. They make watchdog logs actionable
    // without forcing a CPU/GPU sync.
    int64_t launch_time_us       = 0;
    int64_t enqueue_time_us      = 0;
    int64_t result_start_time_us = 0;
    int64_t result_done_time_us  = 0;

    std::atomic<BatchFutureStage> stage{BatchFutureStage::Created};

    // Event placeholders for the executor split. They are optional so the
    // default processAsync() fallback can keep using the synchronous path.
    std::shared_ptr<torch::Event> gpu_done_event;
    std::shared_ptr<torch::Event> rejection_event;
    std::shared_ptr<torch::Event> proposal_event;

    // Keep tensors alive across the engine step boundary until processResults
    // consumes them. Later MTP commits replace this generic list with typed
    // payload fields.
    std::vector<torch::Tensor> tensor_keepalive;

    // bookkeeping_done flips to true once the result thread (or, on the
    // synchronous fallback path, the main thread itself) finishes result
    // processing. asyncStep waits on this before re-entering scheduler while
    // full optimistic scheduling is still gated off.
    std::atomic<bool> bookkeeping_done{false};

    // Status reported by the bookkeeping pass. asyncStep surfaces it on the
    // next iteration so error handling matches the sync path.
    absl::Status bookkeeping_status = absl::OkStatus();
};

using BatchFuturePtr = std::shared_ptr<BatchFuture>;

}  // namespace rtp_llm
