from mobius import losses


def test_ContrastiveLoss(tabular_siamese_record):
    results = set(dir(losses.ContrastiveLoss))
    expected = {"forward"}
    assert expected.issubset(results)

    contrastive_loss = losses.ContrastiveLoss(margin=0.10)
    contrastive_loss.forward(tabular_siamese_record)
    # TODO: assert something about contrastive loss
