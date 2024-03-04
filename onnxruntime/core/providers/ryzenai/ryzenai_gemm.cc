#include "onnxruntime/core/framework.h"
#include "onnxruntime/core/providers/common.h"
#include "ryzenai_gemm.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    GeMM,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    GeMM<float>);

template <typename T>
Status GeMM<T>::Compute(OpKernelContext* context) const override {
  // Access input tensors
  auto* A = context->Input<float>(0);
  auto* B = context->Input<float>(1);

  // Validate input shapes (assuming 2D matrices for simplicity)
  if (A->Shape().NumDimensions() != 2 || B->Shape().NumDimensions() != 2) {
    return Status(common::MakeString("Only 2D matrices supported for MatMul."));
  }

  // Get matrix dimensions
  auto A_rows = A->Shape()[0];
  auto A_cols = A->Shape()[1];
  auto B_rows = B->Shape()[0];
  auto B_cols = B->Shape()[1];

  // Check compatibility for multiplication
  if (A_cols != B_rows) {
    return Status(common::MakeString("Incompatible matrix dimensions for multiplication."));
  }

  // Allocate output tensor
  auto* C = context->Output<float>(0, {A_rows, B_cols});

  // Perform matrix multiplication (basic implementation for illustration)
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < B_cols; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < A_cols; ++k) {
        sum += A->GetData<float>(i * A_cols + k) * B->GetData<float>(k * B_cols + j);
      }
      C->GetData<float>(i * B_cols + j) = sum;
    }
  }

  return onnxruntime::Status::OK();
}
} // namespace MyCustomExecutionProvider
