from typing import Optional, Callable

from torch_geometric.datasets import WebKB
from .DirectedData import DirectedData

def load_directed_real_data(dataset: str='WebKB', root:str = './', name:str = 'Texas',
transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None) -> DirectedData:
	"""The function for WebKB data downloading and convert to DirectedData object

	Args:
		dataset (str, optional) data set name (default: 'WebKB').
	    root (str, optional) path to save the dataset (default: './').
	    name (str, required) the name of the subdataset (default: 'Texas').
		transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
	"""
	if dataset.lower() == 'webkb':
		data = WebKB(root=root,name=name, transform=transform, pre_transform=pre_transform)[0]
	else:
		raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
	directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y)
	return directed_dataset