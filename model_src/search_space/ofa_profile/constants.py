"""
Defines constants for this search space
"""

# PN and MBV3 constants
OFA_W_PN = 1.3

OFA_W_MBV3 = 1.2

# ResNet constants
OFA_RES_STAGE_MAX_N_BLOCKS = (4, 4, 6, 4)
OFA_RES_STAGE_MIN_N_BLOCKS = (2, 2, 4, 2)
OFA_RES_STAGE_BASE_CHANNELS = (256, 512, 1024, 2048)
OFA_RES_WIDTH_MULTIPLIERS = (0.65, 0.8, 1.0)