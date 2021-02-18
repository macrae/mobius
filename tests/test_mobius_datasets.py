from mobius import datasets
from torch import Tensor


def test_unit_test_tabular_siamese_dataset(tabular_siamese_input):
    p1, p2, label = tabular_siamese_input
    assert isinstance(p1, Tensor)
    assert isinstance(p2, Tensor)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)


def test_TabularSiameseDataset():
    results = set(dir(datasets.TabularSiameseDataset))
    expected = {"__getitem__", "_draw", "lbl2rows", "get_items"}
    assert expected.issubset(results)
