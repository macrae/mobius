from mobius import models


def test_TabularSiameseModel():
    results = set(dir(models.TabularSiameseModel))
    expected = {"forward", "encode"}
    assert expected.issubset(results)
