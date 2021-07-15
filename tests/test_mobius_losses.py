import pytest
import torch
from hypothesis import given, settings
from mobius import losses

from strategies import some_normal_distribution, some_siamese_input, some_tabular_siamese_input

# set search strategies behavior to dev (less robust strategies)
settings.load_profile("maximum")


# TODO: refatcor this test...
@given(some_normal_distribution(100))
def test_torch_max(some_normal_distribution):

    for val in some_normal_distribution:
        a = torch.pow(torch.max(torch.tensor([val, 0.])), 2)
        if val < 0.:
            b = torch.tensor([0.])
        else:
            b = torch.pow(torch.tensor([val]), 2)

        assert all(torch.isclose(a.float(), b.float()))


def test_contrastive_loss_class(tabular_siamese_input):
    "Test the ContrastiveLoss object methods"
    results = set(dir(losses.ContrastiveLoss))
    expected = {"forward"}
    assert expected.issubset(results)

    contrastive_loss = losses.ContrastiveLoss(margin=0.0)
    contrastive_loss.forward(tabular_siamese_input)


# TODO: parameterize the contrastive_unit_ pytest fixture with label arg, and combine unit tests
# TODO: take into account batch size and how aggregation over batch is applied
def test_contrastive_loss_unit_pos(siamese_input_pos):
    "Unit test a positive tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward(siamese_input_pos)
    assert loss.numpy() == pytest.approx(1.00, 1e-2)


# TODO: parameterize the contrastive_unit_ pytest fixture with label arg, and combine unit tests
def test_contrastive_loss_unit_neg(siamese_input_neg):
    "Unit test a negative tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward(siamese_input_neg)
    assert loss.numpy() == pytest.approx(0.171, 1e-2)


@given(some_siamese_input())
def test_contrastive_loss_is_positive(some_siamese_input):
    "Property-based test that contrastive loss is never negative"
    p1, p2, y = some_siamese_input.p1, some_siamese_input.p2, some_siamese_input.y
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward((p1, p2, y))
    assert loss.numpy() >= 0.


@settings(max_examples=10_00)
@given(some_tabular_siamese_input())
def test_contrastive_loss_tabular_siamese_model(some_tabular_siamese_input):
    for margin in [0.001, 0.01, 0.1, 1, 10, 20, 50]:
        contrastive_loss = losses.ContrastiveLoss(margin=margin)
        loss = contrastive_loss.forward(some_tabular_siamese_input)
        assert loss.numpy() >= 0.

# TODO:
# 0) using the DrLIM paper, work out by hand the above unit tests (double check implementation)
# 1) add tests for margin (maybe property based?)
