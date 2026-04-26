#pragma once

#include <atomic>
#include <list>
#include <memory>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

// Phase 3.2: per-batch async handoff between the launch (main) thread and the
// bookkeeping (result) thread.
//
// The synchronous NormalEngine::step() path issues schedule -> process ->
// result-bookkeeping back-to-back on the main loop, and the main thread
// blocks on cudaGraphLaunch (~25% of CUDA runtime, see ASYNC_FULL_DESIGN.md
// Phase 3.2). The async path replaces this with:
//   main thread:   schedule N -> launch GPU N -> hand BatchFuture to result
//                  thread -> proceed to schedule N+1 (overlap with N's GPU)
//   result thread: wait gpu_done event -> dispatchDecode / specUpdate /
//                  per-stream housekeeping -> mark bookkeeping_done so the
//                  next main-thread schedule sees consistent state.
//
// The current scaffolding only carries the field set needed to build the
// data path; cudaEvent + folly::Promise wiring lands in an async_opt
// follow-up alongside the executor processAsync / processResults split.
struct BatchFuture {
    // Streams that participated in this batch. Owned references are kept
    // alive until bookkeeping completes so per-stream specUpdate / KV
    // release does not race with main-thread enqueue.
    std::list<GenerateStreamPtr> streams;

    // launchTimeUs records when the main thread submitted the GPU work.
    // Used by metrics + as a coarse staleness guard.
    int64_t launch_time_us = 0;

    // bookkeeping_done flips to true once the result thread (or, on the
    // synchronous fallback path, the main thread itself) finishes
    // dispatchDecode + per-stream updates. asyncStep waits on this before
    // re-entering scheduler so seq_len etc. stay accurate.
    std::atomic<bool> bookkeeping_done{false};

    // Status reported by the bookkeeping pass. asyncStep surfaces it on
    // the next iteration so error handling matches the sync path.
    absl::Status bookkeeping_status = absl::OkStatus();

    // TODO(async_opt):
    //   cudaEvent_t                 gpu_done;
    //   GptModelInputs              model_input;
    //   speculative::SpeculativeSamplerOutput sampler_output;
    //   MergedOutput                draft_extend_output;
    //   folly::Promise<absl::Status> bookkeeping_promise;
};

using BatchFuturePtr = std::shared_ptr<BatchFuture>;

}  // namespace rtp_llm
