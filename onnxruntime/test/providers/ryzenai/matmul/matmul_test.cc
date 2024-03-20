// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include <random>

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

//   // use ryzen EP
//   auto ryzen_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
//     std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
//     execution_providers.push_back(DefaultRyzenAIExecutionProvider());
//     return execution_providers;
//   };
  Ort::SessionOptions so;
  onnxruntime::ProviderOptions options;
  // no real options currently but set a value to make sure it's passed through. requires manual validation.
  options["one"] = "two";
  so.AppendExecutionProvider("RYZENAI", options);
 //so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  const ORTCHAR_T* ort_model_path = ORT_TSTR("C:/Users/alina/ID_comit/onnxruntime/test/testdata/matmul_1_op_version_13.onnx");
  Ort::Session session(*ort_env, ort_model_path, so);
  std::array<float, 3 * 2> input0_data = {7, 7, 7, 7, 7, 7};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add input0
  std::array<int64_t, 2> inputs_shape{3 , 2};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), inputs_shape.data(), inputs_shape.size()));
  ort_input_names.push_back("X");


  // Run session and get outputs
  std::array<const char*, 1> output_names{"Y"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

//      auto ep_vec = DefaultRyzenAIExecutionProvider();
//      RunWithEP(ort_model_path, "MatMulRyzen", std::move(ep_vec), feeds, params);
  // Check output shape.

  Ort::Value& ort_output = ort_outputs[0];
  auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_shape = typeshape.GetShape();
//   auto ep_vec = ryzen_ep();
//     test.Run(so.GetConst().clone(), OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
//              &ep_vec, {});
}

} //test

}// onnxruntime
