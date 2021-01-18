from torch import Tensor

from tabular_siamese import (ContrastiveLoss,
                             TabularSiameseDataset,
                             TabularSiameseModel)


def test_unit_test_tabular_siamese_dataset(unit_test_tabular_siamese_dataset):
    p1, p2, label = unit_test_tabular_siamese_dataset
    assert isinstance(p1, Tensor)
    assert isinstance(p2, Tensor)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)


def test_TabularSiameseDataset():
    results = set(dir(TabularSiameseDataset))
    expected = {"__getitem__", "_draw", "lbl2rows", "get_items"}
    assert expected.issubset(results)
