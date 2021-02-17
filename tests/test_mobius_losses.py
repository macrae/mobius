import pytest
from hypothesis import given, settings
from mobius import losses

from strategies import some_contrastive_record

settings.load_profile("maximum")  # set search strategies behavior to dev (less robust strategies)


# TODO: fill this out more...
def test_ContrastiveLoss(tabular_siamese_record):
    "Test the ContrastiveLoss object methods"
    results = set(dir(losses.ContrastiveLoss))
    expected = {"forward"}
    assert expected.issubset(results)

    contrastive_loss = losses.ContrastiveLoss(margin=0.0)
    contrastive_loss.forward(tabular_siamese_record)


# TODO: parameterize the contrastive_unit_ pytest fixture with label arg, and combine unit tests
def test_ContrastiveLoss_unit_pos(contrastive_unit_pos):
    "Unit test a positive tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward(contrastive_unit_pos)
    assert loss.numpy() == pytest.approx(1.00, 1e-2)


# TODO: parameterize the contrastive_unit_ pytest fixture with label arg, and combine unit tests
def test_ContrastiveLoss_unit_neg(contrastive_unit_neg):
    "Unit test a negative tabular siamese example"
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward(contrastive_unit_neg)
    assert loss.numpy() == pytest.approx(0.171, 1e-2)


@given(some_contrastive_record())
def test_ContrastiveLoss_property_is_positive(some_contrastive_record):
    "Unit test a negative tabular siamese example"
    # import ipdb; ipdb.set_trace()
    p1, p2, y = some_contrastive_record.p1, some_contrastive_record.p2, some_contrastive_record.y
    contrastive_loss = losses.ContrastiveLoss(margin=2.0)
    loss = contrastive_loss.forward((p1, p2, y))
    assert loss.numpy() >= 0.

# TODO:
# 0) using the DrLIM paper, work out by hand the above unit tests (double check implementation)
# 1) add tests for margin (maybe property based?)
