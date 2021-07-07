from typing import Tuple

from fastai.tabular.all import TabularLearner, TabularPandas
from mobius.datasets import TabularSiameseDataset
from torch import Tensor


def test_read_jsonl_bytes():
    list_of_bytes = TabularSiameseDataset.read_jsonl_bytes(
        "./tests/data/test_data.jsonl")
    assert isinstance(list_of_bytes, list)


def test_unit_test_tabular_siamese_dataset(tabular_siamese_input):
    p1, p2, label = tabular_siamese_input
    assert isinstance(p1, Tensor)
    assert isinstance(p2, Tensor)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)


def test_tabular_pandas(init_tabular_pandas):
    tabular_pandas = TabularPandas(**init_tabular_pandas)
    assert isinstance(tabular_pandas, TabularPandas)


def test_tabular_learner_process(to, init_tabular_pandas):
    assert isinstance(to, TabularLearner)

    df = init_tabular_pandas["df"].sample(frac=0.5)
    to_new = to.dls.train_ds.new(df)
    to_new.process()

    assert isinstance(to_new, TabularPandas)


def test_tabular_siamese_dataset(to, init_tabular_pandas):
    results = set(dir(TabularSiameseDataset))
    expected = {"__getitem__", "__len__"}
    assert expected.issubset(results)

    tab_snn_dataset = TabularSiameseDataset(
        csv_file="./tests/data/labels.csv",
        jsonl_file="./tests/data/test_data.jsonl",
        tabular_learner=to)

    assert isinstance(tab_snn_dataset, TabularSiameseDataset)

    # import ipdb; ipdb.set_trace()
    p1, p2, label = tab_snn_dataset.__getitem__(0)
    assert isinstance(p1, Tuple)
    assert isinstance(p2, Tuple)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)
