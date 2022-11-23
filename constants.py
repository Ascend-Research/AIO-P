# general string constants
OOV_TOKEN = "<OOV>"

# DK: string constants used to index batch dict produced by custom data loaders
DK_BATCH_SIZE = "batch_size"
DK_TORCH_GEO_BATCH_DATA_LIST = "torch_geo_batch_data_list"
DK_BATCH_TARGET_TSR = "batch_target_tensor"
DK_BATCH_UNIQUE_STR_ID_SET = "batch_unique_id_set"
DK_BATCH_EDGE_TSR_LIST = "batch_edge_tsr_list"
DK_BATCH_LAST_NODE_IDX_LIST = "batch_last_node_idx_list"
DK_BATCH_CG_REGULAR_IDX = "batch_cg_regular_idx"
DK_BATCH_CG_REGULAR_SHAPES = "batch_cg_regular_shapes"
DK_BATCH_CG_WEIGHTED_IDX = "batch_cg_weighted_idx"
DK_BATCH_CG_WEIGHTED_SHAPES = "batch_cg_weighted_shapes"
DK_BATCH_CG_WEIGHTED_KERNELS = "batch_cg_weighted_kernels"
DK_BATCH_CG_WEIGHTED_BIAS = "batch_cg_weighted_bias"
DK_BATCH_FLOPS = "batch_flops"

# CHKPT: checkpoint dict keys
CHKPT_COMPLETED_EPOCHS = "completed_epochs"
CHKPT_MODEL = "model"
CHKPT_OPTIMIZER = "optimizer"
CHKPT_METADATA = "metadata"
CHKPT_PARAMS = "params"
CHKPT_BEST_EVAL_RESULT = "best_eval_result"
CHKPT_BEST_EVAL_EPOCH = "best_eval_epoch"
CHKPT_PAST_EVAL_RESULTS = "past_eval_results"
CHKPT_ITERATION = "iteration"
CHKPT_BEST_EVAL_ITERATION = "best_eval_iteration"

SEEDS_RAW = [12345,
             1,
             2,
             3,
             4,
             5,
             6,
             7,
             8,
             9]
