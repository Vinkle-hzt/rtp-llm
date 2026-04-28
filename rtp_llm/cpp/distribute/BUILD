load("//:def.bzl", "copts")

package(default_visibility = ["//visibility:public"])

# Pure CPU broadcaster used by tpSyncModelInputs to bypass NCCL's
# cudaDeviceSynchronize stall on the small per-step CPU tensors.
# Star topology over Unix Domain Sockets, intra-node TP only.
cc_library(
    name = "cpu_tp_broadcaster",
    srcs = ["CpuTpBroadcaster.cc"],
    hdrs = ["CpuTpBroadcaster.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
)
