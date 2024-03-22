import onnxruntime
import numpy as np

def run_test():
    onnx_path='C:/Users/alina/ID_comit/onnxruntime/test/testdata/matmul_1_op_version_13.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_path, providers=["RyzenAIExecutionProvider"])
    data = np.full([3, 2], 7, dtype=np.float32)
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name:data})
    print("Output values:", result[0])

if __name__ == "__main__":
    run_test()
