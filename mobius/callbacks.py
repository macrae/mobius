import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from fastai.callback.core import Callback
from sklearn.manifold import TSNE
from torch.nn import Module

from mobius.charts import plot_3d

class EmbeddedKNN(Callback, Module):
    def _knn(self):
        # TODO: write sklearn fit/transform

    def after_epoch(self):

        # embedded training set
        # fit knn=1 on training set
        self.knn = knn

    def forward(self, input, target):
        # y_true = F.one_hot(target, 2).to(torch.float32)
        # y_pred = F.softmax(input, dim=1)

        # tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        # fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        # fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        # precision = tp / (tp + fp + self.epsilon)
        # recall = tp / (tp + fn + self.epsilon)

        # f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        # f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        # return 1 - f1.mean()



class TSNECallback(Callback):
    def after_validate(self):
        plt.clf()
        plt.figure(figsize=(12, 8), dpi=200)
        t = int(time.time())
        
        # TODO: pull this out into a helper function
        # Encode Validation Points
        ################################################################################
        valid_encoded = list()
        for i in range(len(self.dls.valid.dataset.labels)):
            x, _ = self.dls.valid_ds.__getitem__(i)

            # rehsape into mini-batch size 1
            p1 = [t.unsqueeze(0) for t in x[0]]

            # encode the point
            valid_encoded.append(self.model.encode(p1))

        # Encode Training Points
        ################################################################################
        train_encoded = list()
        for i in range(len(self.dls.train.dataset.labels)):
            x, _ = self.dls.train_ds.__getitem__(i)

            # rehsape into mini-batch size 1
            p1 = [t.unsqueeze(0) for t in x[0]]

            # encode the point
            train_encoded.append(self.model.encode(p1))

        y_valid_label = self.dls.valid.dataset.labels["label"]
        y_train_label = self.dls.train.dataset.labels["label"]

        valid_encoded_df = pd.DataFrame(torch.stack(valid_encoded).squeeze())
        train_encoded_df = pd.DataFrame(torch.stack(train_encoded).squeeze())

        # # write encoded space to csv
        # valid_encoded_df.to_csv(f"tsne_{t}_{self.epoch}.csv")

        # UMAP Validation Points
        ################################################################################
        reducer = umap.UMAP()
        embedding_valid_umap = reducer.fit_transform(valid_encoded_df.values)

        xs = embedding_valid_umap[:, 0]
        ys = embedding_valid_umap[:, 1]

        df = pd.DataFrame(zip(xs, ys, y_valid_label), columns=["x", "y", "label"])

        # TODO: add additional umap plot for scatter; also, update legend name.
        sns.scatterplot(data=df, x="x", y="y", hue="label")
        sns.kdeplot(data=df, x="x", y="y", hue="label", fill=True, levels=5, alpha=0.3)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP Projection of Encoded Space', fontsize=12)
        plt.savefig(f"{t}_epoch_{self.epoch}_valid.png",
                    bbox_inches="tight",
                    transparent=True)
        
        # UMAP Validation Points
        ################################################################################
        reducer = umap.UMAP()
        embedding_train_umap = reducer.fit_transform(train_encoded_df.values)

        xs = embedding_train_umap[:, 0]
        ys = embedding_train_umap[:, 1]

        df = pd.DataFrame(zip(xs, ys, y_train_label), columns=["x", "y", "label"])

        # TODO: add additional umap plot for scatter; also, update legend name.
        sns.scatterplot(data=df, x="x", y="y", hue="label")
        sns.kdeplot(data=df, x="x", y="y", hue="label", fill=True, levels=5, alpha=0.3)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP Projection of Encoded Space', fontsize=12)
        plt.savefig(f"{t}_epoch_{self.epoch}_train.png",
                    bbox_inches="tight",
                    transparent=True)

        # # TSNE...
        # # TODO: make this more efficient; configure hyperparams if possible
        # tsne = TSNE(n_components=3, metric="euclidean", n_iter=500)
        # encoded_train_tsne = tsne.fit_transform(valid_encoded_df.values)

        # xs = encoded_train_tsne[:, 0]
        # ys = encoded_train_tsne[:, 1]
        # zs = encoded_train_tsne[:, 2]
        # color = y_valid_label

        # df = pd.DataFrame(zip(xs, ys, zs, color), columns=["x", "y", "z", "c"])

        # # TODO: add the `margin` to the path naem
        # save_path = f"snn_{t}_epoch_{self.epoch}_validation_data.html"
        # plot_3d(df, x="x", y="y", z="z", c="c", symbol="c",
        #         opacity=0.7, save_path=save_path)