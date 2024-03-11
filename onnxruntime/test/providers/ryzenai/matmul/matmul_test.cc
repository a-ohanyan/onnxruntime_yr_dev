// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/quantization_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>
#include <random>

namespace onnxruntime {
namespace test {

TEST(MatmulFloatRyzenTest, MatMulRyzen) {
  OpTester test("MatMul", 20);
  test.AddInput<float>("T1", {4, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  test.AddInput<float>("T2", {3, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  test.AddOutput<float>("T3", {4, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  // use ryzen EP
  auto ryzen_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultRyzenAIExecutionProvider());
    return execution_providers;
  };
  SessionOptions so;
  auto ep_vec = ryzen_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {});
}

} //test

}// onnxruntime
