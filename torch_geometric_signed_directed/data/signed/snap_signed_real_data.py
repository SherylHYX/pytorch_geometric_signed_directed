from typing import Optional, Callable

from torch_geometric.datasets import SNAPDataset
from .SignedData import SignedData

def load_snap_signed_real_data(root:str = './', name:str = 'Epinions',
transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None) -> SignedData:
	"""The function for WebKB data downloading and convert to DirectedData object

	Args:
		dataset (str, optional) data set name (default: 'WebKB').
	    root (str, optional) path to save the dataset (default: './').
	    name (str, required) the name of the subdataset (default: 'Texas').
		transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
	"""
	data = SNAPDataset(root=root,name=name, transform=transform, pre_filter=pre_filter)[0]
	signed_dataset = SignedData(x=data.x,edge_index=data.edge_index,y=data.y)
	return signed_dataset