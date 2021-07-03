import random
from typing import Tuple
from fastai.torch_core import cat

import jsonlines
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.losses import CrossEntropyLossFlat
from fastai.tabular.all import TabularPandas, TabDataLoader
from fastai.torch_basics import params
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset


def write_jsonl(df, path) -> None:
    df.to_json(path, orient='records', lines=True)


class TabularSiameseDataset(Dataset):
    """Siamese Pairs for Tabular dataset."""

    # TODO: avoid passing the tabular learner to the Dataset, just the attr of the learner
    def __init__(self, csv_file, jsonl_file, tabular_learner):
        self.labels = pd.read_csv(csv_file, index_col=0)
        self.y_name = tabular_learner.dls.train.y_names[0]
        self.c = len(set(self.labels[self.y_name]))
        self.jsonl_file = jsonl_file
        self.jsonl_cache = self.cache_jsonl(jsonl_file)
        self.to = tabular_learner
        self.cat_names = tabular_learner.dls.cat_names
        self.cont_names = tabular_learner.dls.cont_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # load line from json lines file
        row_i = self.load_jsonl(self.jsonl_file, span=self.jsonl_cache[idx])
        row_j, same = self._draw(idx)

        # apply pre-processing
        row_i_cats, row_i_conts = self.apply_procs(row_i)
        row_j_cats, row_j_conts = self.apply_procs(row_j)

        return (row_i_cats.reshape(-1), row_i_conts.reshape(-1)), (row_j_cats.reshape(-1), row_j_conts.reshape(-1)), torch.Tensor([int(same)]).squeeze()

    def _draw(self, idx):
        cls = self.labels.iloc[idx, ][self.y_name]
        same = random.random() < 0.5
        if not same:
            cls = random.choice(
                [label for label in set(self.labels.values.flatten()) if label != cls])
        return self.lbl2rows(cls), int(not same)

    def lbl2rows(self, cls):
        idxs = [i for i, label in enumerate(
            self.labels.values) if label == cls]
        random_idx = random.choice(idxs)
        return self.load_jsonl(self.jsonl_file, span=self.jsonl_cache[random_idx])

    def apply_procs(self, row) -> Tuple[tensor]:
        # load row from dict to dataframe
        to = self.to.dls.train_ds.new(pd.DataFrame(row, index=[0]))

        # apply pre-processing pipeline
        to.process()

        # select cats & conts from dataframe
        cats, conts = to.items[self.cat_names].values, to.items[self.cont_names].values
        return tensor(cats).long(), tensor(conts).float()

    @staticmethod
    def cache_jsonl(jsonl_file) -> dict:
        with open(jsonl_file, 'r') as f:
            chars = f.read()
            spans = list()
            span = [0]
            for i, c in enumerate(chars):
                if c == "\n":
                    spans.append(span + [i])
                    span = [i + 1]
        return spans

    @staticmethod
    def load_jsonl(jsonl_file, span) -> dict:
        with open(jsonl_file, 'r') as f:
            chars = f.read()
            text = chars[span[0]: span[1]].replace("false", "False").replace("true", "True")
            return eval(text)
