#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
namespace onnxruntime {
// Information needed to construct RyzenAI execution providers.
struct RyzenAIExecutionProviderInfo {
  bool create_arena{true};

  explicit RyzenAIExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  RyzenAIExecutionProviderInfo() = default;
};

// using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&,
//                                       std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class RyzenAIExecutionProvider : public IExecutionProvider {
 public:
  // delay_allocator_registration = true is used to allow sharing of allocators between different providers that are
  // associated with the same device
  explicit RyzenAIExecutionProvider(const RyzenAIExecutionProviderInfo& info);

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  //std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  RyzenAIExecutionProviderInfo info_;
//   std::vector<FuseRuleFn> fuse_rules_;
};

// Registers all available RyzenAI kernels
Status RegisterKernels(KernelRegistry& kernel_registry);
}; //namespace onnxruntime
