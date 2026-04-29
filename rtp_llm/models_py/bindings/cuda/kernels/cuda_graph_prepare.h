#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace rtp_llm {

constexpr int kMaxCudaGraphPrepareFillRegions = 32;

struct CudaGraphPrepareFillRegion {
    int32_t* ptr   = nullptr;
    int64_t  count = 0;
    int32_t  value = 0;
};

struct CudaGraphPrepareFillParams {
    int32_t                    region_count = 0;
    CudaGraphPrepareFillRegion regions[kMaxCudaGraphPrepareFillRegions];
};

void invokeCudaGraphPrepareFill(CudaGraphPrepareFillParams params, cudaStream_t stream);

}  // namespace rtp_llm
