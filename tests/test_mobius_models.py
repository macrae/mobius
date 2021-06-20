from mobius.models import TabularSiameseModel


def test_tabular_siamese_model_class():
    results = set(dir(TabularSiameseModel))
    expected = {"forward", "encode"}
    assert expected.issubset(results)


def test_tabular_siamese_model_train():
    # TODO: load tabular dataset and train a tabular siamese model with seed.
    assert True
