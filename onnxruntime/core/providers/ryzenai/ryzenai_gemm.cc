#include "ryzenai_gemm.h"
namespace onnxruntime {


ONNX_RYZENAI_OPERATOR_TYPED_KERNEL(
    MatMul,
    20,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ryzenai::MatMul<float>);
ONNX_RYZENAI_OPERATOR_TYPED_KERNEL(
    MatMul,
    20,
    int,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int>()),
    ryzenai::MatMul<int>);

namespace ryzenai{

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* context) const {
  // Access input tensors
  const auto* A = context->Input<Tensor>(0);
  const auto* B = context->Input<Tensor>(1);

  // Validate input shapes (assuming 2D matrices for simplicity)
  if (A->Shape().NumDimensions() != 2 || B->Shape().NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Only 2D matrices supported");
  }

  // Get matrix dimensions
  auto A_rows = A->Shape()[0];
  auto A_cols = A->Shape()[1];
  auto B_rows = B->Shape()[0];
  auto B_cols = B->Shape()[1];

  // Check compatibility for multiplication
  if (A_cols != B_rows) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inner dimension should match");
  }

  // Allocate output tensor
  auto* C = context->Output(0, {A_rows, B_cols});
  auto a_data = A->Data<T>();
  auto b_data = B->Data<T>();
  auto c_data = C->MutableData<T>();
  // Perform matrix multiplication (basic implementation for illustration)
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < B_cols; ++j) {
      T sum = 0;
      for (int k = 0; k < A_cols; ++k) {
        sum += a_data[i * A_cols + k] * b_data[k * B_cols + j];
      }
      c_data[i * B_cols + j] = sum;
    }
  }
  printf("ryzenai matmul \n");
  return onnxruntime::Status::OK();
}
  }
} // namespace MyCustomExecutionProvider
