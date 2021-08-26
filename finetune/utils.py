import os
import logging
import numpy as np


def set_logging_config(logdir):
    """
    set logging configuration
    :param logdir: directory put logs
    :return: None
    """
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])


def get_pruning_mask(score, pruning_ratio, pruning_method):
    """
    :param score: numpy.ndarray, the score used for pruning. Elements with lower scores
                  are pruned, i.e. are 0 in the mask.
    :param pruning_ratio: a float.
    :param pruning_method: select betwenn 'layerwise' and 'global'
    """
    if pruning_method == 'layerwise':
        num_layers, num_elements_per_layer = score.shape
        num_pruned_elements_per_layer = round(num_elements_per_layer * pruning_ratio)
        pruning_mask = np.ones_like(score).astype(np.int32)
        sorted_indices = np.argsort(score, axis=-1)
        for l in range(num_layers):
            pruned_indices = sorted_indices[l, :num_pruned_elements_per_layer]
            for idx in pruned_indices:
                pruning_mask[l, idx] = 0
        return pruning_mask.tolist()
    else:
        num_elements = score.size
        num_pruned_elements = round(num_elements * pruning_ratio)
        score_flat = score.reshape(-1)
        pruning_mask_flat = np.ones_like(score_flat).astype(np.int32)
        sorted_flat_indices = np.argsort(score_flat)
        for idx in sorted_flat_indices[:num_pruned_elements]:
            pruning_mask_flat[idx] = 0
        return pruning_mask_flat.reshape(score.shape).tolist()
