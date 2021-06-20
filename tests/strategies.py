import random
from collections import namedtuple
from typing import Tuple

import hypothesis.stateful as stateful
import hypothesis.strategies as st
import numpy as np
import torch
from torch import tensor

########################################################################################
#                                 General Strategies                                   #
########################################################################################

# TODO: fix this search strategy, it returns null labels
some_label: stateful.SearchStrategy = st.sampled_from([0, 1])


@st.composite
def some_binary_label(draw) -> int:
    "Search strategy for a binary classification label"
    return random.choice([0, 1])


@st.composite
def some_normal_distribution(draw, n) -> float:
    "Search strategy for a normally distributed sample; mean=0.0, std=1.0"
    mu, sigma = 0.0, 1.0  # mean, standard deviation
    return np.random.normal(mu, sigma, n)


########################################################################################
#                               Siamese Net Strategies                                 #
########################################################################################


@st.composite
def some_tensor(draw, n) -> tensor:
    "Search strategy for a tensor of uniformly distributed values"
    return tensor([random.uniform(-1_000, 1_000) for x in range(0, n)])


# let's give the
SiameseInput = namedtuple("SiameseInput", ["p1", "p2", "y"])


@st.composite
def some_siamese_input(draw) -> SiameseInput:
    "Search strategy for a simple siamese network input"

    # the length of tensors in the siamese pair
    n = draw(st.integers(min_value=100, max_value=10_00))

    # draw the siamese network points and label
    p1 = draw(some_tensor(n))
    p2 = draw(some_tensor(n))
    y = draw(some_binary_label())

    return SiameseInput(p1, p2, y)

########################################################################################
#                         fast.ai Tabular Siamese Strategies                           #
########################################################################################

# TODO: this is good for testing TabularSiamaseModel mocking stuff...
@st.composite
def some_cats(draw, n) -> tensor:
    # TODO: make this integer encoded
    return tensor([random.uniform(1, 500) for x in range(0, n)]).long()


@st.composite
def some_conts(draw, n) -> tensor:
    # TODO: draw from gaussian distribution - mean 0, std 1
    return tensor(draw(some_normal_distribution(n))).float()


@st.composite
def some_tabular_point(draw, n) -> tensor:
    return (draw(some_cats(n)), draw(some_conts(n)))


@st.composite
def some_tabular_siamese_input(draw) -> Tuple[tensor]:
    n = draw(st.integers(min_value=3, max_value=10))
    p1 = torch.cat(draw(some_tabular_point(n)))
    p2 = torch.cat(draw(some_tabular_point(n)))
    label = draw(some_binary_label())
    y = torch.Tensor([label]).squeeze()
    return p1, p2, y
