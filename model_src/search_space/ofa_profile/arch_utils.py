from utils.math_utils import make_divisible

def get_final_channel_size(C_base, w):
    return make_divisible(C_base * w, divisor=8)


def get_final_channel_sizes(C_list, w):
    return [get_final_channel_size(c, w) for c in C_list]


