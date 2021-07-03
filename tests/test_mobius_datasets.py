from typing import Tuple

from fastai.tabular.all import TabularLearner, TabularPandas
from mobius.datasets import TabularSiameseDataset
from torch import Tensor


def test_cache_jsonl():
    spans = TabularSiameseDataset.cache_jsonl(jsonl_file="./tests/data/test_data.jsonl")
    assert isinstance(spans, list)
    assert spans[0] == [0, 49]
    assert spans[1] == [50, 99]
    assert spans[2] == [100, 149]


def test_load_jsonl():
    spans = TabularSiameseDataset.cache_jsonl("./tests/data/test_data.jsonl")
    for span in spans:
        jsonl = TabularSiameseDataset.load_jsonl("./tests/data/test_data.jsonl", span)
        assert isinstance(jsonl, dict)


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

    p1, p2, label = tab_snn_dataset.__getitem__(0)
    assert isinstance(p1, Tuple)
    assert isinstance(p2, Tuple)
    assert all(isinstance(t, Tensor) for t in p1)
    assert all(isinstance(t, Tensor) for t in p2)
    assert isinstance(label, Tensor)
