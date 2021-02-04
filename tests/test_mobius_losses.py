import pytest
from mobius import losses


def test_ContrastiveLoss(tabular_siamese_record):
    "Test the ContrastiveLoss object methods"
    results = set(dir(losses.ContrastiveLoss))
    expected = {"forward"}
    assert expected.issubset(results)

    contrastive_loss = losses.ContrastiveLoss(margin=0.0)
    contrastive_loss.forward(tabular_siamese_record)


def test_ContrastiveLoss_unit_pos(tabular_siamese_unit_pos):
    "Unit test a positive tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=0.0)
    loss = contrastive_loss.forward(tabular_siamese_unit_pos)
    assert loss.numpy() == pytest.approx(0.00, 1e-2)


def test_ContrastiveLoss_unit_neg(tabular_siamese_unit_neg):
    "Unit test a negative tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=0.0)
    loss = contrastive_loss.forward(tabular_siamese_unit_neg)
    assert loss.numpy() == pytest.approx(40.00, 1e-2)

# TODO:
# 0) using the DrLIM paper, work out by hand the above unit tests (double check implementation)
# 1) add tests for margin (maybe property based?)
