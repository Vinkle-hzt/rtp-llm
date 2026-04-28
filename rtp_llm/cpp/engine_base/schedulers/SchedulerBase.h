#pragma once

#include <list>
#include <vector>
#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

class SchedulerBase {
public:
    virtual ~SchedulerBase() {}
    virtual absl::Status                   enqueue(const GenerateStreamPtr& stream)                    = 0;
    virtual std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<GenerateStreamPtr>& streams) = 0;
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> schedule()                                    = 0;

    // Phase 3.3: conservative-KV scheduling variant for the async (Phase 3.2)
    // pipeline. The async path schedules step N+1 before step N's
    // specUpdate has run, so seq_len isn't yet authoritative. Conservative
    // variants assume the maximum possible accept_len (propose_step + 1) for
    // each in-flight speculative stream, then the result thread releases the
    // surplus blocks once accept_len is known.
    //
    // Default implementation: identity passthrough. Subclasses that support
    // the Phase 3.2 ultimate path (NormalEngine::asyncStep without the
    // bookkeeping_done.wait() hop) override this with the speculative-aware
    // allocator. Callers that don't enable the ultimate async path can
    // continue to use schedule() unchanged.
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> scheduleConservative(int /*propose_step*/) {
        return schedule();
    }
    // F5a async scheduling guard: scheduleConservative() may run before the
    // previous result future has committed GenerateDone/Error. After awaiting
    // that future, callers use this hook to consume terminal running-stream
    // events without doing another reserve-step allocation.
    virtual absl::Status refreshRunningStreams(std::list<GenerateStreamPtr>& /*streams*/) {
        return absl::OkStatus();
    }
    virtual absl::Status stop()             = 0;
    virtual bool         empty()            = 0;
    virtual int64_t      lastScheduleTime() = 0;
    virtual int64_t      onflightStreams()  = 0;

    virtual std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() {
        return {};
    }
    virtual std::vector<EngineScheduleInfo::TaskInfo> runningTaskList() {
        return {};
    }
    virtual void updateSchedulerInfo(const std::string& scheduler_info) {}
};

}  // namespace rtp_llm
