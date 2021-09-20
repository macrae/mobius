import torch
from fastai.tabular.all import TabularModel
from torch.nn import Module


class TabularSiameseModel(Module):
    def __init__(self, encoder):
        super(TabularSiameseModel, self).__init__()
        assert isinstance(encoder, TabularModel)
        self.encoder = encoder

    def forward(self, x):
        emb_1, emb_2 = self.encoder(*x[0]), self.encoder(*x[1])
        return emb_1, emb_2

    def encode(self, x):
        return self.encoder(*x)


class TabularSiameseModelBinaryCrossEntropy(Module):
    def __init__(self, encoder, head):
        super(TabularSiameseModelBinaryCrossEntropy, self).__init__()
        assert isinstance(encoder, TabularModel)
        self.encoder, self.head = encoder, head

    def forward(self, x):
        ftrs = torch.cat([self.encoder(*x[0]), self.encoder(*x[1])], dim=1)
        return self.head(ftrs)
