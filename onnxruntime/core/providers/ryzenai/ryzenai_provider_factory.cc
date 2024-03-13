#include <memory>

#include "core/providers/ryzenai/ryzenai_execution_provider.h"
#include "core/providers/ryzenai/ryzenai_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct RyzenAIProviderFactory : IExecutionProviderFactory {
  RyzenAIProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~RyzenAIProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> RyzenAIProviderFactory::CreateProvider() {
  RyzenAIExecutionProviderInfo info;
  info.create_arena = false;
  return std::make_unique<RyzenAIExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> RyzenAIProviderFactoryCreator::Create(const ProviderOptions& provider_options) {
  return std::make_shared<onnxruntime::RyzenAIProviderFactory>(false);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_RyzenAI, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::RyzenAIProviderFactoryCreator::Create({}));
  return nullptr;
}
