// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/ryzenai/ryzenai_execution_provider.h"
#include <absl/base/config.h>
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};

RyzenAIExecutionProvider::RyzenAIExecutionProvider(const RyzenAIExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kRyzenAIExecutionProvider, OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, 10)}, info_{info} {}

std::vector<AllocatorPtr> RyzenAIExecutionProvider::CreatePreferredAllocators() {

  AllocatorCreationInfo device_info{[](int) { return std::make_unique<CPUAllocator>(); },
                                    DEFAULT_CPU_ALLOCATOR_DEVICE_ID, false};

  return std::vector<AllocatorPtr>{CreateAllocator(device_info)};
}


class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRyzenAIExecutionProvider, kOnnxDomain, 1, float, GeMM);

// !!PLEASE READ BELOW!! Following that, add new entries above this comment

/*  *** IMPORTANT! ***
 NEVER update a versioned entry to change the start or end version. These MUST be treated as immutable.
   i.e. if the macro has 'VERSIONED' in it, do not modify that entry

 When updating a declaration to add a new version of an operator there are 2 simple steps:

   1. There should be a non-versioned entry for that latest version. Update this to be versioned.
      Note that the end version is inclusive, so the end value will be one less than the operator's new opset version.
   2. Add a new non-versioned entry for the new opset.

 e.g. Say opset 13 is being added, and we need to update Add. The most recent change to Add was in opset 7 so it
      should have an un-versioned registration in the opset 7 section like this:

     class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRyzenAIExecutionProvider, kOnnxDomain, 7, float, Add);

   Step 1 is to change that to add 'VERSIONED_' to the macro and add an end version of 12 as the new opset is 13:
     class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRyzenAIExecutionProvider, kOnnxDomain, 7, 12, float, Add);

   Step 2 is to create a new un-versioned entry in the opset 13 sections:
     class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRyzenAIExecutionProvider, kOnnxDomain, 13, float, Add);

 The process is the same for TYPED and untyped kernels - just repeat for each type when updating the typed entries.

 The changes below in the registrations using BuildKernelCreateInfo are essentially the same. Update existing
 registration to use the VERSIONED_ macro, add end version, add new un-versioned entry in the section for the new
 opset.

 To double-check what versions an operator should have registrations for see
 https://github.com/onnx/onnx/blob/main/docs/Operators.md
*****/

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterOnnxOperatorKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRyzenAIExecutionProvider, kOnnxDomain, 1,
                                                                          float, GeMM)>
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}


Status RegisterKernels(KernelRegistry& kernel_registry) {
  ORT_RETURN_IF_ERROR(RegisterOnnxOperatorKernels(kernel_registry));
  return Status::OK();
}

KernelRegistryAndStatus GetKernelRegistryStatus() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> RyzenAIExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetKernelRegistryStatus();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

// std::unique_ptr<IDataTransfer> RyzenAIExecutionProvider::GetDataTransfer() const {
//   return std::make_unique<CPUDataTransfer>();
// }

}  // namespace onnxruntime
