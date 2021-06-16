from tests.conftest import tabular_siamese_input
from fastai.tabular.all import TabularPandas
from mobius.datasets import TabularSiameseDataset
from torch import Tensor


def test_unit_test_tabular_siamese_dataset(tabular_siamese_input):
    p1, p2, label = tabular_siamese_input
    assert isinstance(p1, Tensor)
    assert isinstance(p2, Tensor)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)


def test_tabular_siamese_dataset(init_tabular_pandas):
    results = set(dir(TabularSiameseDataset))
    expected = {"__getitem__", "_draw", "lbl2rows", "get_items"}
    assert expected.issubset(results)

    tabular_pandas = TabularPandas(**init_tabular_pandas)
    assert isinstance(tabular_pandas, TabularPandas)

    tabular_siamese_pandas = TabularSiameseDataset(**init_tabular_pandas)
