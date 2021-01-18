import torch
from fastai.tabular.all import TabularModel
from torch.nn import Module


class TabularSiameseModel(Module):
    def __init__(self, encoder, head):
        super(TabularSiameseModel, self).__init__()
        assert isinstance(encoder, TabularModel)
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(*x1), self.encoder(*x2)], dim=1)
        return self.head(ftrs)

    def encode(self, x):
        return self.encoder(*x)
