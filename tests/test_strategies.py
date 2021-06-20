from typing import Tuple

from hypothesis import given, settings
from torch import Tensor

import strategies as st

settings.load_profile("maximum")


@given(st.some_tabular_siamese_input())
def test_some_tabular_siamese_input(some_tabular_siamese_input):

    # unpack the siamase input
    p1, p2, y = some_tabular_siamese_input

    # check types
    assert isinstance(some_tabular_siamese_input, Tuple)

    assert isinstance(p1, Tensor)
    assert isinstance(p1[0], Tensor)
    assert isinstance(p1[1], Tensor)

    assert isinstance(p2, Tensor)
    assert isinstance(p2[0], Tensor)
    assert isinstance(p2[1], Tensor)

    assert isinstance(y, Tensor)
