#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

inline std::string pdDebugEnvRankString() {
    const char* env_names[] = {"RANK", "WORLD_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "LOCAL_RANK", "SLURM_PROCID"};
    for (const char* name : env_names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            std::ostringstream oss;
            oss << name << "=" << value;
            return oss.str();
        }
    }
    return "env_rank=unknown";
}

inline std::string pdDebugRankString() {
    std::ostringstream oss;
    oss << pdDebugEnvRankString() << " pid=" << getpid();
    return oss.str();
}

inline std::string pdDebugRankString(const ParallelismConfig& parallelism_config) {
    std::ostringstream oss;
    oss << "world_rank=" << parallelism_config.world_rank << " tp_rank=" << parallelism_config.tp_rank
        << " dp_rank=" << parallelism_config.dp_rank << " local_rank=" << parallelism_config.local_rank << " "
        << pdDebugRankString();
    return oss.str();
}

inline const char* pdBoolString(bool value) {
    return value ? "true" : "false";
}

inline const char* pdCacheGroupTypeString(int32_t value) {
    switch (value) {
        case 0:
            return "LINEAR";
        case 1:
            return "FULL";
        default:
            return "UNKNOWN";
    }
}

inline void pdMtpDebugLog(const std::string& component, const std::string& message) {
    std::cout << "[PD_MTP_DEBUG][" << component << "][" << pdDebugRankString() << "] " << message << std::endl;
}

inline void
pdMtpDebugLog(const std::string& component, const ParallelismConfig& parallelism_config, const std::string& message) {
    std::cout << "[PD_MTP_DEBUG][" << component << "][" << pdDebugRankString(parallelism_config) << "] " << message
              << std::endl;
}

}  // namespace rtp_llm
