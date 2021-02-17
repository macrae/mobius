import random
from collections import namedtuple
from typing import Tuple

import hypothesis.stateful as stateful
import hypothesis.strategies as st
import numpy as np
import torch
from torch import tensor

ContrastiveRecord = namedtuple("ContrastiveRecord", ["p1", "p2", "y"])

# binary classification labels
LABELS = [0, 1]
some_label: stateful.SearchStrategy = st.sampled_from(LABELS)

# a normally distributed sample, mean=0.0, std=1.0
@st.composite
def some_normally_distributed_value(draw, n) -> float:
    mu, sigma = 0.0, 1.0  # mean, standard deviation
    return np.random.normal(mu, sigma, n)


@st.composite
def some_point(draw, n) -> tensor:
    return tensor([random.uniform(-1_000, 1_000) for x in range(0, n)])


@st.composite
def some_contrastive_record(draw) -> ContrastiveRecord:
    n = draw(st.integers(min_value=100, max_value=10_00))
    p1 = draw(some_point(n))
    p2 = draw(some_point(n))
    y = draw(some_label)
    return ContrastiveRecord(p1, p2, y)


def some_cats(draw, n) -> tensor:
    return tensor([random.uniform(1, 500) for x in range(0, n)])

def some_cats(draw, n) -> tensor:
    return tensor([random.uniform(1, 500) for x in range(0, n)])

def some_tabular_point(draw, n) -> Tuple[tensor]:
    return (
        torch.cat((
            tensor([1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
                    2,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   1,
                    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
                    1,  16,  36, 267, 145,   7,   2,   1,   1,   1,   1,   1,   1,   1,
                    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]),
            tensor([-1.1155, -1.1123,  0.4383, -0.0690, -0.0563,  0.8495, -0.2410, -0.5648,
                    -0.6675,  0.5577,  0.3845, -0.0335,  0.0766, -0.6007, -0.6971, -0.8765,
                    -0.1549, -1.9072, -0.1248, -0.4690, -0.0971, -0.5916, -1.4794, -1.4851,
                    -0.1631, -0.3891, -0.2723, -0.2381, -0.4545, -0.3706, -0.0554, -0.2190,
                    -0.0911, -0.1687, -0.8812, -0.9850, -0.7922, -0.1687, -0.8792, -0.9902,
                    -0.8008, -0.7724, -0.7845, -0.7878, -0.2968, -0.7018, -0.2727, -0.3229,
                    -1.0393, -1.0504, -0.1261, -0.5045, -0.8001, -0.5200, -0.5234, -1.1787,
                    -1.1904, -0.1685, -0.5963, -0.8548, -0.5794, -0.5739, -1.4884, -1.7928,
                    -0.0397, -0.4046, -0.2693, -0.0129, -0.1262, -0.5546, -0.1236])
        )),
