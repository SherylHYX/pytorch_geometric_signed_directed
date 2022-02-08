from torch_geometric.datasets import WebKB
from .DirectedData import DirectedData

def load_WebKB(root:str = './', name:str = 'Texas') -> DirectedData:
	"""The function for WebKB data downloading and convert to DirectedData object

	Args:
	    root (str, optional) path to save the dataset (default: './')
	    name (str, required) the name of the subdataset (default: 'Texas')
	"""
	data = WebKB(root=root,name=name)[0]
	directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y)
	return directed_dataset