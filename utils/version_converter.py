import onnx
from onnx import version_converter
def add_const_value_infos_to_graph(graph : onnx.GraphProto):
    inputs = {i.name for i in graph.input}
    existing_info = {vi.name: vi for vi in graph.input}
    for init in graph.initializer:
        # Check it really is a constant, not an input
        if init.name in inputs:
            continue

        # The details we want to add
        elem_type = init.data_type
        shape = init.dims

        # Get existing or create new value info for this constant
        vi = existing_info.get(init.name)
        if vi is None:
            vi = graph.input.add()
            vi.name = init.name

        # Even though it would be weird, we will not overwrite info even if it doesn't match
        tt = vi.type.tensor_type
        if tt.elem_type == onnx.TensorProto.UNDEFINED:
            tt.elem_type = elem_type
        if not tt.HasField("shape"):
            # Ensure we set an empty list if the const is scalar (zero dims)
            tt.shape.dim.extend([])
            for dim in shape:
                tt.shape.dim.add().dim_value = dim

    # Handle subgraphs
    for node in graph.node:
        for attr in node.attribute:
            # Ref attrs refer to other attrs, so we don't need to do anything
            if attr.ref_attr_name != "":
                continue

            if attr.type == onnx.AttributeProto.GRAPH:
                add_const_value_infos_to_graph(attr.g)
            if attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    add_const_value_infos_to_graph(g)
    return



model = onnx.load("../onnxruntime/test/testdata/matmul_1.onnx")
add_const_value_infos_to_graph(model.graph)
converted_model = version_converter.convert_version(model, 20)
onnx.save(converted_model, "./matmul_1_op_version_20.onnx")
