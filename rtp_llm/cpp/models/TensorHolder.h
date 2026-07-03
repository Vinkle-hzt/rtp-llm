#pragma once

#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

struct TensorHolder {
    static constexpr size_t kReleasedHoldRounds = 2;

    std::vector<torch::Tensor>             tensors;
    std::queue<std::vector<torch::Tensor>> clear_tensors;

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    template<typename T,
             typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, torch::Tensor>
                                         && !std::is_same_v<std::decay_t<T>, TensorHolder>>>
    void hold(const T& object) {
        objects.emplace_back(std::make_unique<HeldObject<std::decay_t<T>>>(object));
    }

    void release() {
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels are not freed until the third release point.
        clear_tensors.push(std::move(tensors));
        tensors.clear();
        clear_objects.push(std::move(objects));
        objects.clear();
        while (clear_tensors.size() > kReleasedHoldRounds) {
            clear_tensors.pop();
        }
        while (clear_objects.size() > kReleasedHoldRounds) {
            clear_objects.pop();
        }
    }

private:
    struct HeldObjectBase {
        virtual ~HeldObjectBase() = default;
    };

    template<typename T>
    struct HeldObject: HeldObjectBase {
        explicit HeldObject(const T& object): object(object) {}

        T object;
    };

    std::vector<std::unique_ptr<HeldObjectBase>>             objects;
    std::queue<std::vector<std::unique_ptr<HeldObjectBase>>> clear_objects;
};

}  // namespace rtp_llm
