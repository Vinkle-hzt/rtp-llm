#pragma once

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/devices/OpData.h"

namespace rtp_llm {

struct SpecMetricsCollectors {
    RtpLLMExecutorMetricsCollector          executor_collector;
    RtpLLMTokenPSMetricsCollector           tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector sp_engine_collector;

    bool not_skip = false;
};

}  // namespace rtp_llm