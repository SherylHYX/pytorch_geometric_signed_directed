from typing import Optional, Union

import torch
import numpy as np


class Prob_Imbalance_Loss(torch.nn.Module):
    r"""An implementation of the probablistic imbalance loss function from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Args:
        F (int or NumPy array, optional) - Number of pairwise imbalance socres to consider, or the meta-graph adjacency matrix.
    """

    def __init__(self, F: Optional[Union[int, np.ndarray]] = None):
        super(Prob_Imbalance_Loss, self).__init__()
        if isinstance(F, int):
            self.sel = F
        elif F is not None:
            K = F.shape[0]
            self.sel = 0
            for i in range(K-1):
                for j in range(i+1, K):
                    if (F[i, j] + F[j, i]) > 0:
                        self.sel += 1

    def forward(self, P: torch.FloatTensor, A: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                K: int, normalization: str = 'vol_sum', threshold: str = 'sort') -> torch.FloatTensor:
        """Making a forward pass of the probablistic imbalance loss function from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.
        Arg types:
            * **prob** (PyTorch FloatTensor) - Prediction probability matrix made by the model
            * **A** (PyTorch FloatTensor, can be sparse) - Adjacency matrix A
            * **K** (int) - Number of clusters
            * **normalization** (str, optional) - normalization method:

                'vol_sum': Normalized by the sum of volumes, the default choice.

                'vol_max': Normalized by the maximum of volumes.        

                'vol_min': Normalized by the minimum of volumes.   

                'plain': No normalization, just CI.   
            * **threshold**: (str, optional) normalization method:

                'sort': Picking the top beta imbalnace values, the default choice.

                'std': Picking only the terms 3 standard deviation away from null hypothesis.  

                'naive': No thresholding, suming up all K*(K-1)/2 terms of imbalance values.  

        Return types:
            * **loss** (torch.Tensor) - loss value, roughly in [0,1].
        """
        assert normalization in ['vol_sum', 'vol_min', 'vol_max',
                                 'plain'], 'Please input the correct normalization method name!'
        assert threshold in [
            'sort', 'std', 'naive'], 'Please input the correct threshold method name!'

        device = A.device
        # avoid zero volumn to be denominator
        epsilon = torch.FloatTensor([1e-8]).to(device)
        # first calculate the probabilitis volumns for each cluster
        vol = torch.zeros(K).to(device)
        for k in range(K):
            vol[k] = torch.sum(torch.matmul(
                A + torch.transpose(A, 0, 1), P[:, k:k+1]))
        second_max_vol = torch.topk(vol, 2).values[1] + epsilon
        result = torch.zeros(1).to(device)
        imbalance = []
        if threshold == 'std':
            imbalance_std = []
        for k in range(K-1):
            for l in range(k+1, K):
                w_kl = torch.matmul(P[:, k], torch.matmul(A, P[:, l]))
                w_lk = torch.matmul(P[:, l], torch.matmul(A, P[:, k]))
                if (w_kl-w_lk).item() != 0:
                    if threshold != 'std' or np.power((w_kl-w_lk).item(), 2)-9*(w_kl+w_lk).item() > 0:
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk)
                        imbalance.append(curr)
                    else:  # below-threshold values in the 'std' thresholding scheme
                        if normalization == 'vol_sum':
                            curr = torch.abs(w_kl-w_lk) / \
                                (vol[k] + vol[l] + epsilon) * 2
                        elif normalization == 'vol_min':
                            curr = torch.abs(
                                w_kl-w_lk)/(w_kl + w_lk)*torch.min(vol[k], vol[l])/second_max_vol
                        elif normalization == 'vol_max':
                            curr = torch.abs(
                                w_kl-w_lk)/(torch.max(vol[k], vol[l]) + epsilon)
                        elif normalization == 'plain':
                            curr = torch.abs(w_kl-w_lk)/(w_kl + w_lk)
                        imbalance_std.append(curr)
        imbalance_values = [curr.item() for curr in imbalance]
        if threshold == 'sort':
            # descending order
            ind_sorted = np.argsort(-np.array(imbalance_values))
            for ind in ind_sorted[:int(self.sel)]:
                result += imbalance[ind]
            # take negation to be minimized
            return torch.ones(1, requires_grad=True).to(device) - result/self.sel
        elif len(imbalance) > 0:
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance))
        elif threshold == 'std':  # sel is 0, then disregard thresholding
            return torch.ones(1, requires_grad=True).to(device) - torch.mean(torch.FloatTensor(imbalance_std))
        else:  # nothing has positive imbalance
            return torch.ones(1, requires_grad=True).to(device)
