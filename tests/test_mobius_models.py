from torch import Tensor

from tabular_siamese import (ContrastiveLoss,
                             TabularSiameseDataset,
                             TabularSiameseModel)


def test_TabularSiameseModel():
    results = set(dir(TabularSiameseModel))
    expected = {"forward", "encode"}
    assert expected.issubset(results)
