from torch_geometric_signed_directed.data import load_snap_signed_real_data, SignedData

def test_signed_datasets():
    # Test signed real data
    signed_dataset = load_snap_signed_real_data(root='./tmp_data/', name='soc-epinions1')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    signed_dataset = load_snap_signed_real_data(root='./tmp_data/', name='soc-slashdot0811')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    signed_dataset = load_snap_signed_real_data(root='./tmp_data/', name='soc-slashdot0922')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    return