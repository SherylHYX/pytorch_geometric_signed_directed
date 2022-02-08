from torch_geometric_signed_directed.data.directed import Datasets

def test_datasets():
	"""
	Testing load_dataset()
	"""
	directed_dataset = load_dataset()
	assert isinstance(load_dataset(), DirectedData)
	return