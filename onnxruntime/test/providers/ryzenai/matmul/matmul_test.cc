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
	const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata\\matmul_1_op_version_20.onnx");
	std::cout << "Begin : Input creation.." << std::endl;
	Ort::SessionOptions so;
	Ort::Session session(*ort_env, ort_model_path, so);
	std::cout << "End  : Session creation.." << std::endl;
	auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//TBD : input_shape[0] = 1;

	TensorShape input_shape_x{input_shape};
	RandomValueGenerator generator;
	std::vector<float> input_x = generator.Uniform<float>(input_shape_x.GetDims(),
			7.0, 7.0);
	OrtValue ml_value_x;
	CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);

	NameMLValMap feeds;
	feeds.insert(std::make_pair("X", ml_value_x));

	EPVerificationParams params;
	params.ep_node_assignment = ExpectedEPNodeAssignment::All;
	//params.fp32_abs_err = 0.0002f;
	//params.graph_verifier = &verify;
	auto ep_vec = DefaultRyzenAIExecutionProvider();

	RunWithEP(ort_model_path, "MatMulRyzen", std::move(ep_vec), feeds, params);
}

} //test

}// onnxruntime


