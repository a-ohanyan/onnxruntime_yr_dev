#pragma once

#include "core/framework/op_kernel.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace ryzenai{
template <typename T>
class MatMul : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const;

 private:
 TensorShape b_shape_;
 IAllocatorUniquePtr<void> packed_b_;
};
}

} // namespace onnxruntime
