// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/ryzenai/ryzenai_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(RyzenAIExecutionProviderTest, MetadataTest) {
  RyzenAIExecutionProviderInfo info;
  auto provider = std::make_unique<RyzenAIExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_EQ(provider->GetOrtDeviceByMemType(OrtMemTypeDefault).Type(), OrtDevice::CPU);
}
}  // namespace test
}  // namespace onnxruntime
