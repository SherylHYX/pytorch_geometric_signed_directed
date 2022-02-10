from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData, link_class_split, node_class_split

def test_directed_datasets():
    """
    Testing load_dataset()
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
    directed_dataset = load_directed_real_data(dataset='cora_ml', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='citeseer', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikics', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/', name='chameleon')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/', name='squirrel')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    return

def test_link_split():
    """
    Testing link_split
    """
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
    datasets = link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'direction')
    datasets = link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'existence')
    datasets = link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'all')
    return