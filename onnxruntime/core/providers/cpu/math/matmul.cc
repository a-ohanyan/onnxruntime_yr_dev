// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

// opset 9 supports more types
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, uint32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9,
    12,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int64_t, uint64_t>()),
    MatMul<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, uint32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    13,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int64_t, uint64_t>()),
    MatMul<int64_t>);

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // Using DataRaw as int32_t/uint32_t and int64_t/uint64_t share a common
  // operator body.
  const auto* a_data = reinterpret_cast<const T*>(a->DataRaw());
  const auto* b_data = reinterpret_cast<const T*>(b->DataRaw());
  auto* y_data = reinterpret_cast<T*>(y->MutableDataRaw());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    math::MatMul<T>(
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        a_data + helper.LeftOffsets()[i],
        b_data + helper.RightOffsets()[i],
        y_data + helper.OutputOffsets()[i],
        thread_pool);
  }

  return Status::OK();
}
#if defined(__aarch64__) && defined(__linux__)
bool GemmPackBBfloat16(AllocatorPtr& alloc,
                       const Tensor& tensor_b,
                       bool trans_b,
                       IAllocatorUniquePtr<void>& packed_b,
                       size_t& packed_b_size,
                       TensorShape& b_shape) {
  // Only handle the common case of a 2D weight matrix. Additional matrices
  // could be handled by stacking the packed buffers.
  if (tensor_b.Shape().NumDimensions() != 2) {
    return false;
  }

  b_shape = tensor_b.Shape();

  const size_t K = trans_b ? static_cast<size_t>(b_shape[1]) : static_cast<size_t>(b_shape[0]);
  const size_t N = trans_b ? static_cast<size_t>(b_shape[0]) : static_cast<size_t>(b_shape[1]);

  packed_b_size = MlasSBGemmPackBSize(N, K);
  if (packed_b_size == 0) {
    return false;
  }

  packed_b = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
  auto* packed_b_data = packed_b.get();

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_b_data, 0, packed_b_size);
  MlasSBGemmConvertPackB(N,
                         K,
                         tensor_b.Data<float>(),
                         trans_b ? K : N,
                         packed_b_data);
  return true;
}
#endif

Status MatMul<float>::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // only pack Matrix B
  if (input_idx == 1) {
    size_t packed_b_size;
#if defined(__aarch64__) && defined(__linux__)
    size_t dim1 = 0;
    size_t dim2 = 0;
    TensorShape b_shape = tensor.Shape();

    if (b_shape.NumDimensions() == 2) {
      dim1 = static_cast<size_t>(b_shape[0]);
      dim2 = static_cast<size_t>(b_shape[1]);
    }

    if (use_fastmath_mode_ && (trans_b_attr_ == 0) && ((dim1 * dim2) >= kFastMathModeKernelsizeThreshold)) {
      is_packed = GemmPackBBfloat16(alloc, tensor, trans_b_attr_ != 0, packed_b_, packed_b_size, b_shape_);
    } else
#endif
    {
      is_packed = GemmPackBFp32(alloc, tensor, trans_b_attr_ != 0, packed_b_, packed_b_size, b_shape_);
    }

    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (is_packed && share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size);
    }
  }
  return Status::OK();
}

Status MatMul<float>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                int input_idx,
                                                /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }

  return Status::OK();
}

Status MatMul<float>::Compute(OpKernelContext* context) const {

  std::cout << "In CPU MatMul : Implementation copied from Ryzenai matmul ... " << std::endl;
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = packed_b_ ? nullptr : context->Input<Tensor>(1);
  const auto& b_shape = B ? B->Shape() : b_shape_;

  // Validate input shapes (assuming 2D matrices for simplicity)
  if (A->Shape().NumDimensions() != 2 || b_shape.NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Only 2D matrices supported");
  }

  // Get matrix dimensions
  auto A_rows = A->Shape()[0];
  auto A_cols = A->Shape()[1];
  auto B_rows = b_shape[0];
  auto B_cols = b_shape[1];

  // Check compatibility for multiplication
  if (A_cols != B_rows) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inner dimension should match");
  }

  // Allocate output tensor
  auto* C = context->Output(0, {A_rows, B_cols});
  auto a_data = A->Data<float>();
  const auto b_data = B ? B->Data<float>() : (float*)packed_b_.get();
  auto c_data = C->MutableData<float>();

  // Perform matrix multiplication (basic implementation for illustration)
  for (int i = 0; i < A_rows; ++i) {
    for (int j = 0; j < B_cols; ++j) {
      float sum = 0;
      for (int k = 0; k < A_cols; ++k) {
        sum += a_data[i * A_cols + k] * b_data[k * B_cols + j];
      }
      c_data[i * B_cols + j] = sum;
    }
  }
  return onnxruntime::Status::OK();
}

}  // namespace onnxruntime
