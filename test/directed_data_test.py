from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData

def test_datasets():
	"""
	Testing load_dataset()
	"""
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Cornell')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Wisconsin')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_directed_real_data(dataset='citation', root='./tmp_data/', name='Cora_ML')
	assert isinstance(directed_dataset, DirectedData)
	directed_dataset = load_directed_real_data(dataset='citation', root='./tmp_data/', name='CiteSeer')
	assert isinstance(directed_dataset, DirectedData)
	return