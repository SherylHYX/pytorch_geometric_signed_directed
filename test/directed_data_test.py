from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData

def test_directed_datasets():
	"""
	Testing load_directed_real_data()
	"""
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
	assert isinstance(directed_dataset, DirectedData)
	assert directed_dataset.is_directed
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Cornell')
	assert isinstance(directed_dataset, DirectedData)
	assert directed_dataset.is_directed
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Wisconsin')
	assert isinstance(directed_dataset, DirectedData)
	assert directed_dataset.is_directed
	return