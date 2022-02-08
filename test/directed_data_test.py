from torch_geometric_signed_directed.data import load_WebKB, DirectedData

def test_datasets():
	"""
	Testing load_dataset()
	"""
	directed_dataset = load_WebKB(root='./tmp_data/', name='Texas')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_WebKB(root='./tmp_data/', name='Cornell')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_WebKB(root='./tmp_data/', name='Wisconsin')
	assert isinstance(directed_dataset, DirectedData)
	return