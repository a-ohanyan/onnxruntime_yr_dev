// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include <random>

#include "core/common/logging/logging.h"
#include "core/common/span_utils.h"
#include "core/framework/utils.h"
#include "core/graph/graph.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/quantization_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;
#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

namespace onnxruntime {
namespace test {

TEST(MatmulFloatRyzenTest, MatMulRyzen) {
//   OpTester test("MatMul", 20);
//   test.AddInput<float>("T1", {4, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
//   test.AddInput<float>("T2", {3, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
//   test.AddOutput<float>("T3", {4, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

   // use ryzen EP
   auto ryzen_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
     std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
     execution_providers.push_back(DefaultRyzenAIExecutionProvider());
     return execution_providers;
   };
  Ort::SessionOptions so;
  onnxruntime::ProviderOptions options;
  // no real options currently but set a value to make sure it's passed through. requires manual validation.
  options["one"] = "two";
  so.AppendExecutionProvider("RYZENAI", options);
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata\\matmul_1_op_version_20.onnx");
  Ort::Session session(*ort_env, ort_model_path, so);

    // dirty hack to access the underlying InferenceSession but don't know a better way.
    const OrtSession* ort_session = session;
    const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);

    bool have_ryzenai_ep = false;

    for (const auto& provider : s->GetRegisteredProviderTypes()) {
      if (provider == kRyzenAIExecutionProvider) {
        have_ryzenai_ep = true;
        break;
      }
    }
    ASSERT_TRUE(have_ryzenai_ep) << "ryzenai EP was not found in registered providers for session.";

    RandomValueGenerator generator;
    TensorShape input_shape_x{3, 2};
    std::vector<float> input_x = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};//generator.Uniform<float>(input_shape_x.GetDims(), 0, 128);

    OrtValue ml_value_x;
    CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);

    NameMLValMap feeds;
    feeds.insert(std::make_pair("X", ml_value_x));

    TensorShape input_weights{2, 1};
    std::vector<float> input_w = {2, 3};//generator.Uniform<float>(input_weights.GetDims(), 0, 128);

    OrtValue ml_value_w;
    CreateMLValue<float>(input_weights.GetDims(), input_w.data(), OrtMemoryInfo(), &ml_value_w);
    feeds.insert(std::make_pair("W", ml_value_w));
  //EPVerificationParams params;
  //params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  //params.fp32_abs_err = 0.0002f;
  //params.graph_verifier = &verify;
    auto ep_vec = ryzen_ep();
    RunAndVerifyOutputsWithEP(ort_model_path, "MatMulRyzen", std::move(ep_vec), feeds);

//     test.Run(so.GetConst().clone(), OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
//              &ep_vec, {});
}

} //test

}// onnxruntime


