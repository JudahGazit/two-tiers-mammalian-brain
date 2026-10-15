import numpy as np
from dataloaders.brain_net import BrainNet


def core_periphery_average_weight(brain_net: BrainNet, expected=0.3):
    avg_weight = np.nansum(brain_net.fibers, 0) / np.nansum(brain_net.adjacency, 0)
    coreness = avg_weight > np.nanquantile(avg_weight, 1 - expected)
    return coreness
