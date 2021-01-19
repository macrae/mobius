import random

import torch
import torch.nn.functional as F
from fastai.losses import CrossEntropyLossFlat
from fastai.tabular.all import TabularModel, TabularPandas
from fastai.torch_basics import params
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset


class TabularSiameseDataset(Dataset):

    def __init__(self, tabular_pandas):
        try:
            assert type(tabular_pandas) is TabularPandas
        except Exception:
            print("tabular_pandas must be TabularPandas")
        self.tabular_pandas = tabular_pandas

    def __getitem__(self, i):
        # get the ith item
        row1_cats, row1_conts = (
            tensor(
                self.tabular_pandas.train.dataloaders().cats.iloc[i].values),
            tensor(
                self.tabular_pandas.train.dataloaders().conts.iloc[i].values)
        )

        # get some jth item and a label indicating sameness wrt the ith
        row2, same = self._draw(i)
        row2_cats, row2_conts = row2[0], row2[1]

        return (
            (row1_cats.long(), row1_conts.float()),
            (row2_cats.long(), row2_conts.float()),
            torch.Tensor([int(same)]).squeeze().long()
        )

    def __len__(self): return len(self.tabular_pandas)

    def _draw(self, i):
        cls = self.tabular_pandas.train.iloc[i][self.tabular_pandas.targ.columns[0]]
        same = random.random() < 0.5
        if not same:
            cls = random.choice(
                [label for label in set(self.tabular_pandas.y) if label != cls])
        return self.lbl2rows(cls), int(same)

    def lbl2rows(self, cls):
        train_df = self.tabular_pandas.train
        idx = random.choice(train_df[train_df.y == cls].index.values)
        cats, conts = tensor(train_df.dataloaders().cats.loc[idx].values), tensor(
            train_df.dataloaders().conts.loc[idx].values)
        return cats, conts

    def get_items(self, i, j, show=False):
        x1, _, _ = self.__getitem__(i)
        x1_cls = self.tabular_pandas.vocab[self.tabular_pandas.train.y.iloc[i]]

        x2, _, _ = self.__getitem__(j)
        x2_cls = self.tabular_pandas.vocab[self.tabular_pandas.train.y.iloc[j]]

        return x1, x2, x1_cls == x2_cls


# TODO: implement siamese vs. triplet HERE...
class TabularTripletDataset(torch.utils.data.Dataset):
    """A Dataset for generating Siamese record sets... sumthing, sumthing....
    """

    def __init__(self, tabular_pandas):
        try:
            assert type(tabular_pandas) is TabularPandas
        except Exception:
            print("tabular_pandas must be TabularPandas")
        self.tabular_pandas = tabular_pandas

    def __getitem__(self, i):

        # p1
        row1_cats = torch.Tensor(
            self.tabular_pandas.train.dataloaders().cats.iloc[i].values)
        row1_conts = torch.Tensor(
            self.tabular_pandas.train.dataloaders().conts.iloc[i].values)

        # p2 (same as p1)
        row2_cats, row2_conts = self._draw(i, same=True)

        # neg (list of points different from p1)
        negs = [self._draw(i, same=False) for x in range(5)]

        # TODO: param eps
        neg_dists = [
            F.pairwise_distance(
                torch.stack(row1_cats, row1_conts), torch.stack(negs[i]), eps=1e-6, keepdim=False)
            for i in range(len(negs))]

        # index of the closest negative
        min_idx = torch.argmin(torch.stack(
            [x.mean() for x in neg_dists])).item()

        neg = [(cats.long(), conts.float())
               for cats, conts in neg_dists[min_idx]]
        # target = torch.Tensor([int(True)]).squeeze().long()

        return (row1_cats.long(), row1_conts.float()), (row2_cats.long(), row2_conts.float()), neg

    def __len__(self): return len(self.tabular_pandas)

    def _draw(self, i, same: bool):
        cls = self.tabular_pandas.train.iloc[i][self.tabular_pandas.targ.columns[0]]
        if not same:
            cls = random.choice(
                [label for label in set(self.tabular_pandas.y) if label != cls])
        return self.lbl2rows(cls)

    def lbl2rows(self, cls):
        train_df = self.tabular_pandas.train
        idx = random.choice(train_df[train_df.y == cls].index.values)
        cats = torch.Tensor(train_df.dataloaders().cats.loc[idx].values)
        conts = torch.Tensor(train_df.dataloaders().conts.loc[idx].values)
        return cats, conts

    def get_items(self, i, j, show=False):
        x1, _, _ = self.__getitem__(i)
        x1_cls = self.tabular_pandas.vocab[self.tabular_pandas.train.y.iloc[i]]

        x2, _, _ = self.__getitem__(j)
        x2_cls = self.tabular_pandas.vocab[self.tabular_pandas.train.y.iloc[j]]

        if show:
            df = pd.DataFrame(zip(torch.cat(x1).numpy(), torch.cat(
                x2).numpy()), columns=["x1", "x2"])
            cls = pd.DataFrame(zip([x1_cls], [x2_cls]), columns=["x1", "x2"])

            df = pd.concat([cls, df], axis=0)
            feature_names = self.tabular_pandas.cat_names + self.tabular_pandas.cont_names
            target_name = self.tabular_pandas.targ.columns[0]

            df["Features"] = [target_name] + feature_names
            df.set_index("Features", inplace=True)
            print(df)

        return x1, x2, x1_cls == x2_cls
