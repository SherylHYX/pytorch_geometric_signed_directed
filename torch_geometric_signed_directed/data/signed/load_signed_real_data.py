from typing import Optional, Callable, Union, List

from .SDGNN_real_data import SDGNN_real_data
from .SSSNET_real_data import SSSNET_real_data
from .MSGNN_real_data import MSGNN_real_data
from .SignedData import SignedData


def load_signed_real_data(dataset: str = 'epinions', root: str = './tmp_data/',
                          transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                          train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                          test_size: Union[int, float] = None, seed_size: Union[int, float] = None,
                          train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                          test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                          seed: List[int] = [], data_split: int = 10, sparsify_level: float=1) -> SignedData:
    """The function for real-world signed data downloading and convert to SignedData object.

    Arg types:
        * **dataset** (str, optional) - data set name (default: 'epinions').
        * **root** (str, optional) - The path to save the dataset (default: './').
        * **transform** (callable, optional) - A function/transform that takes in an \
            :obj:`torch_geometric.data.Data` object and returns a transformed \
            version. The data object will be transformed before every access. (default: :obj:`None`)
        * **pre_transform** (callable, optional) - A function/transform that takes in \
            an :obj:`torch_geometric.data.Data` object and returns a \
            transformed version. The data object will be transformed before \
            being saved to disk. (default: :obj:`None`)
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.  
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)
        * **sparsify_level** (float, optional) - the density of the graph, a value between 0 and 1, for MSGNN data only. Default: 1.

    Return types:
        * **data** (Data) - The required data object.
    """
    if dataset.lower() in ['bitcoin_otc', 'bitcoin_alpha', 'wiki', 'slashdot', 'epinions']:
        data = SDGNN_real_data(
            name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() in ['sp1500', 'rainfall', 'sampson', 'wikirfa', 'ppi'] or dataset[:8].lower() == 'fin_ynet':
        data = SSSNET_real_data(
            name=dataset, root=root, transform=transform, pre_transform=pre_transform)[0]
    elif dataset[:4].lower() == 'fill':
        data = MSGNN_real_data(
            name=dataset, root=root, transform=transform, pre_transform=pre_transform, sparsify_level=sparsify_level)[0]
    else:
        raise NameError(
            'Please input the correct data set name instead of {}!'.format(dataset))
    signed_dataset = SignedData(
        edge_index=data.edge_index, edge_weight=data.edge_weight, init_data=data)
    if train_size is not None or train_size_per_class is not None:
        signed_dataset.node_split(train_size=train_size, val_size=val_size,
                                  test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
                                  val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
                                  seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)
    return signed_dataset
