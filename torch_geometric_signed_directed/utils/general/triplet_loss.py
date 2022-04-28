from typing import Union
import random

import torch
import numpy as np


def triplet_loss_node_classification(y: Union[np.array, torch.Tensor], Z: torch.FloatTensor,
                                     n_sample: int, thre: float):
    r""" Triplet loss function for node classification.

    Arg types:
        * **y** (np.array or torch.Tensor) - Node labels.
        * **Z** (torch.FloatTensor) - Embedding matrix for nodes.
        * **n_sample** (int) - Number of samples.
        * **thre** (float) - Threshold value for the triplet differences to be counted.

    Return types:
        * **loss** (torch.FloatTensor) - Loss value.
    """
    nclass = int(y.max() - y.min() + 1)
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    all_nodes_array = np.arange(Z.shape[0])
    n_sample_class = max((int)(n_sample / nclass), 32)
    loss = 0
    for i in range(nclass):
        randInds1 = random.choices(all_nodes_array[y == i], k=n_sample_class)
        randInds2 = random.choices(all_nodes_array[y == i], k=n_sample_class)

        feats1 = Z[randInds1]
        feats2 = Z[randInds2]
        randInds_dif = random.choices(
            all_nodes_array[y != i], k=n_sample_class)

        feats_dif = Z[randInds_dif]

        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss
