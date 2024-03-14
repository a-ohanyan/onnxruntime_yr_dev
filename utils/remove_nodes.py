import onnx_graphsurgeon as gs
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_file_path", required=True, type=str)
parser.add_argument("--remove_nodes", required=True, type=str)
parser.add_argument("--out_name", required=True, type=str)
args = parser.parse_args()

model = onnx.load(args.onnx_file_path)
graph = gs.import_onnx(model)

#for i in graph.nodes:
#    print(i.name)

for init in model.graph.node:
    if init.op_type == args.remove_nodes:
        print (init.op_type)
        print (init.name)
        if(init.name == "/tulrv6/embeddings/Add_2_output_0_QuantizeLinear"):
            continue
        remove_node = [
            node for node in graph.nodes if node.name == init.name 
        ][0]

        # Get the input node of the fake node
        # Node provides i() and o() functions that can optionally
        # be provided an index (default is 0)
        # These serve as convenience functions for the alternative,
        # which would be to fetch the input/output
        # tensor first, then fetch the input/output node of the tensor.
        # For example, node.i() is equivalent to node.inputs[0].inputs[0]
        inp_node = remove_node.i()
        
        # Reconnect the input node to the output tensors of the fake node,
        # so that the first identity node in the example graph now
        # skips over the fake node.
        inp_node.outputs = remove_node.outputs
        remove_node.outputs.clear()
    
        # Remove the fake node from the graph completely
        graph.cleanup()

onnx.save(gs.export_onnx(graph), args.out_name)
