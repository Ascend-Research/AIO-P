import argparse
from model_src.comp_graph.tf_comp_graph import ComputeGraph
from model_src.predictor.gpi_family_data_manager import FamilyDataManager

"""
Simple script to demonstrate the API for visualizing Compute Graphs by printing a few random ones
Run "python visualize_cgs.py -h" for help
Output file names contain the family, the number of the CG (randomly sampled, not terribly relevant)
    as well as the decimal values of the CG's accuracy score, e.g., 95.5% is represented as "acc955"
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-family", type=str, default="ofa_pn_sem_seg_coco_ind",
                        help="name of architecture family")
    parser.add_argument("-num_cgs", type=int, default=1,
                        help="number of random cgs to print")
    parser.add_argument("-output_dir", type=str, default=None,
                        help="whether to output printed cgs to a specific directory")
    params = parser.parse_args()

    dm = FamilyDataManager([params.family])
    family2sets = dm.get_regress_train_dev_test_sets(0, 0)

    cgs = []
    for partition in family2sets:
        cgs += partition

    for i in range(params.num_cgs):
        cg = cgs[i][0]
        print(cg)
        cg.gviz_visualize(view=False, output_dir=params.output_dir, 
                          filename="{}_acc{}".format(cg.name, str(cgs[i][1]).replace("0.", "")))
