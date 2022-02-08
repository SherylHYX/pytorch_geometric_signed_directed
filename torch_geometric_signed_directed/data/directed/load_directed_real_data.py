from torch_geometric.datasets import WebKB, CitationFull
from .DirectedData import DirectedData

def load_directed_real_data(dataset: str='WebKB', root:str = './', name:str = 'Texas') -> DirectedData:
	"""The function for WebKB data downloading and convert to DirectedData object

	Args:
		dataset (str, optional) data set name (default: 'WebKB').
	    root (str, optional) path to save the dataset (default: './').
	    name (str, required) the name of the subdataset (default: 'Texas').
	"""
	if dataset.lower() == 'webkb':
		data = WebKB(root=root,name=name)[0]
	elif dataset.lower() == 'citation':
		data = CitationFull(root=root,name=name)[0]
	else:
		raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
	directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y)
	return directed_dataset