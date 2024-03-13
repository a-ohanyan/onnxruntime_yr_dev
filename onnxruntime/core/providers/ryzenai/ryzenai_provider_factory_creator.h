#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
struct RyzenAIProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ProviderOptions& provider_options);
};
}
