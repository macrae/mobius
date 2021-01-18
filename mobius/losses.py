import torch
import torch.nn.functional as F
from torch.nn import Module


class ContrastiveLoss(Module):
    """Takes embeddings of two records and a target label == 1 if samples are from the same class
    and label == 0 otherwise
    """

    def __init__(self, margin=0.10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, ops, size_average=True):
        p1, p2, label = ops[0], ops[1], ops[2]

        # distance between positive tensors
        euclidean_distance = F.pairwise_distance(
            p1.reshape(1, -1), p2.reshape(1, -1), keepdim=False)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class F1ScoreLoss(Module):
    """Calculate F1 score. Can work with gpu tensors
    """

    def __init__(self, epsilon=1e-7):
        super(F1ScoreLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        y_true = F.one_hot(target, 2).to(torch.float32)
        y_pred = F.softmax(input, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
