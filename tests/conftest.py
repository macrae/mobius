"""pytest configuration file"""
import pandas as pd
import pytest
import torch
from fastai.data.core import DataLoaders, range_of
from fastai.metrics import F1Score, Precision, Recall, accuracy
from fastai.tabular.all import (Categorify, CategoryBlock, FillMissing,
                                Normalize, RandomSplitter, TabDataLoader,
                                TabularPandas, tabular_config, tabular_learner)
from hypothesis import HealthCheck, Verbosity, settings
from torch import tensor

# register hypothesis search strategy options
settings.register_profile(
    "maximum",
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
    max_examples=300
)

settings.register_profile(
    "ci",
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
    max_examples=50
)
settings.register_profile(
    "dev",
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
    max_examples=10
)
settings.register_profile(
    "debug",
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
    max_examples=3,
    verbosity=Verbosity.verbose,
)


# TODO: combine siamese_input_pos & siamese_input_neg into siamese_input(label: int)
@pytest.fixture
def siamese_input_pos():
    "A single, static example of input into a siamese network for the palmers penguins df"
    return (
        tensor([1, 0]),
        tensor([0, 1]),
        tensor(0)
    )


@pytest.fixture
def siamese_input_neg():
    "A single, static example of input into a siamese network for the palmers penguins df"
    return (
        tensor([1, 0]),
        tensor([0, 1]),
        tensor(1)
    )


@pytest.fixture
def tabular_siamese_input():
    "A single, static example of input into a siamese network for the palmers penguins df"
    import torch
    from torch import tensor

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

        torch.cat((
            tensor([1,   1,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1,   2,   2,
                    2,   2,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,
                    1,   1,   1,   1,   1,   1,   2,   1,   2,   2,   2,   1,   1,   1,
                    1,  15,   8, 102,  41,   6,   1,   2,   1,   1,   1,   1,   1,   1,
                    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]),
            tensor([-0.1558, -0.0206,  0.4824, -0.0690, -0.0563,  0.9815, -0.1191,  0.5792,
                    1.8147,  0.7748,  0.3845, -0.3263, -0.6351,  0.2850, -0.4629, -0.2969,
                    -0.5617,  0.1767, -0.5504,  1.1134,  1.4220,  2.0691,  1.2294,  1.2923,
                    -0.1631,  3.6763, -0.2723, -0.2381, -0.4545, -0.3706, -0.0554, -0.2190,
                    1.7613, -0.1687,  2.2116,  1.3842,  1.4823, -0.1687,  2.1337,  1.4056,
                    1.5612,  1.1147,  1.9652,  1.9293,  4.4502,  1.9539,  4.0257,  2.7324,
                    1.4440,  1.5309, -0.1261,  2.8677,  1.7353,  2.1948,  1.2704,  1.2004,
                    1.3700, -0.1685,  2.5621,  1.6272,  2.0446,  0.8310, -1.4884, -1.7928,
                    -0.0397, -0.4046, -0.2693, -1.6486,  6.0291,  0.3345,  0.3269])
        )),

        tensor(0)
    )


@pytest.fixture
def init_tabular_pandas():
    init = {}

    # dataframe
    X = [
        [0.2, 0.1, "A", "B", 1],
        [0.3, 0.2, "B", "B", 0],
        [0.1, 0.4, "C", "A", 0],
        [0.2, 0.1, "A", "B", 1],
        [0.3, 0.2, "B", "B", 0],
        [0.1, 0.4, "C", "A", 0],
        [0.2, 0.1, "A", "B", 1],
        [0.3, 0.2, "B", "B", 0],
        [0.1, 0.4, "C", "A", 0],
        [0.2, 0.1, "A", "B", 1],
        [0.3, 0.2, "B", "B", 0],
        [0.1, 0.4, "C", "A", 0],
    ]
    df = pd.DataFrame(X, columns=["val_0", "val_1", "str_0", "str_1", "label"])
    init["df"] = df

    # tabular pre-processing procs
    init["procs"] = [FillMissing, Categorify, Normalize]

    # categorical feature names
    init["cat_names"] = ["str_0", "str_1"]

    # continuous feature names
    init["cont_names"] = ["val_0", "val_1"]

    # target names
    init["y_names"] = ["label"]

    # some fast.ai thing :shrug:
    init["y_block"] = CategoryBlock()

    # dataset splitter func
    init["splits"] = RandomSplitter(valid_pct=0.10)(range_of(df))

    return init


@pytest.fixture
def to(init_tabular_pandas):
    tabular_pandas = TabularPandas(**init_tabular_pandas)

    # create tabular dataloader
    trn_dl = TabDataLoader(
        tabular_pandas.train,
        bs=2,
        shuffle=True,
        drop_last=True)

    val_dl = TabDataLoader(
        tabular_pandas.valid,
        bs=2,)

    dls = DataLoaders(trn_dl, val_dl)

    # load the tabular_pandas data through the tabular_learner
    layers = [32, 16]

    # tabular learner configuration
    config = tabular_config(ps=[0.2, 0.1], embed_p=0.1)

    return tabular_learner(
        dls,
        layers=layers,
        config=config,
        metrics=[accuracy,
                 Precision(average='macro'),
                 Recall(average='macro'),
                 F1Score(average='macro')])


@pytest.fixture
def palmer_penguins_df():
    "The palmers penguins dataset"
    from palmerpenguins import load_penguins
    return load_penguins()
