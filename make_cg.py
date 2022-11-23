import argparse
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I

"""
Simple script to generate a Compute Graph from a Tensorflow protobuf (.pb) file.
We provide sample .pb files for EfficientNet-b0 and ResNet18 with the .pkl and .json files
Some words of caution:
    - The provided CG interface expects the network to return a single output tensor, 
        e.g., as is typically done in image classification. Networks that return multiple outputs
        or objects that are not tf tensors may produce errors.
    - The options for tensor dimensions, specifically h_in and w_in, are for normalizing
        the CG node features for prediction. The visualized CG .png file will not reflect
        these options as the existing dimensions in the .pb file will override it.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-pb_file", type=str, required=True,
                        help="Name of pb file to process.")
    parser.add_argument("-cg_name", type=str, default="MyNewCG",
                        help="String to set the name field of the compute graph. Used for saving a visualization.")
    parser.add_argument("-h_in", type=int, default=224,
                        help="Image height used to normalize features.")
    parser.add_argument("-w_in", type=int, default=224,
                        help="Image width used to normalize features.")
    parser.add_argument("-c_in", type=int, default=3,
                        help="Image channels.")
    parser.add_argument("-output_dir", type=str, default=None,
                        help="Whether to output printed cgs to a specific directory.")
    params = parser.parse_args()

    op2i = OP2I().build_from_file()

    new_cg = ComputeGraph(name=params.cg_name, 
                          H=params.h_in, W=params.w_in, C_in=params.c_in)
    new_cg.build_from_pb(params.pb_file, op2i, oov_threshold=0.)

    # Now the CG object is actually built, and you can do with it what you please.
    # For now, we'll just print it out.

    new_cg.gviz_visualize(view=False, filename=params.cg_name, output_dir=params.output_dir)
