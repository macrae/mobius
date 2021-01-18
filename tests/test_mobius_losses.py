from torch import Tensor

from tabular_siamese import (ContrastiveLoss,
                             TabularSiameseDataset,
                             TabularSiameseModel)


def test_ContrastiveLoss(unit_test_tabular_siamese_dataset):
    results = set(dir(ContrastiveLoss))
    expected = {"forward"}
    assert expected.issubset(results)

    contrastive_loss = ContrastiveLoss(margin=0.10)
    contrastive_loss.forward(unit_test_tabular_siamese_dataset)
    # TODO: assert something about contrastive loss
