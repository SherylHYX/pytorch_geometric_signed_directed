from DirectedData import DirectedData
from torch_geometric.datasets import WebKB

def load_dataset(dataset:str = 'webkb', root:str = './', name:str = 'Texas'):
	"""The function for data downloading and convert to DirectedData object

	Args:
	    dataset (str, optional) dataset name (default: 'WebKB')
	    root (str, optional) path to save the dataset (default: './')
	    name (str, required) the name of the subdataset (default: 'Texas')
	"""
	if dataset.lower() == 'webkb':
		data = WebKB(root=root,name=name)[0]
		directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y)
	return directed_dataset