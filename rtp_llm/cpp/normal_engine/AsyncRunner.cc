#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <thread>

namespace rtp_llm {

AsyncRunner::AsyncRunner(torch::Stream stream): stream_(stream), event_(stream.device_type()) {
    thread_ = std::thread([this] { workerLoop(); });
}

AsyncRunner::~AsyncRunner() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        shutdown_ = true;
    }
    cv_task_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void AsyncRunner::launch(std::function<void()> fn) {
    RTP_LLM_PROFILE_SCOPE("async_runner.launch");
    at::ThreadLocalState tls_state;
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_; });
        pending_task_ = Task{std::move(fn), std::move(tls_state)};
        task_done_    = false;
    }
    cv_task_.notify_one();
}

void AsyncRunner::sync(const torch::Stream& wait_stream) {
    RTP_LLM_PROFILE_SCOPE("async_runner.sync");
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    lk.unlock();
    event_.block(wait_stream);
}

void AsyncRunner::workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_task_.wait(lk, [this] { return pending_task_.has_value() || shutdown_; });
            if (shutdown_ && !pending_task_.has_value()) {
                return;
            }
            task = std::move(*pending_task_);
            pending_task_.reset();
        }

        {
            at::ThreadLocalStateGuard tls_guard(task.tls_state);
            RTP_LLM_PROFILE_SCOPE("async_runner.thread");
            cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(stream_));
            task.fn();
            // Block worker CPU until stream_ drains: torch caching allocator handed
            // main-stream allocations memory the worker stream was still reading.
            stream_.synchronize();
            event_.record(stream_);
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            task_done_ = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
