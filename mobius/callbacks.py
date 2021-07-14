import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from fastai.callback.core import Callback
from sklearn.manifold import TSNE

from mobius.charts import plot_3d


class TSNECallback(Callback):
    def after_validate(self):
        # # half of the time run the callback
        # if random.random() < 0.5:
        #     return
        plt.clf()
        plt.figure(figsize=(12, 8), dpi=200)
        t = int(time.time())
        valid_encoded = list()
        for i in range(len(self.dls.valid.dataset.labels)):
            p, _, _ = self.dls.valid_ds.__getitem__(i)

            # rehsape into mini-batch size 1
            p = p[0].reshape(1, -1), p[1].reshape(1, -1)

            # encode the household into output embedding space
            p_encode = self.model.encode(p)
            valid_encoded.append(p_encode)

        y_valid_label = self.dls.valid.dataset.labels["label"]
        valid_encoded_df = pd.DataFrame(torch.stack(valid_encoded).squeeze())

        # write encoded space to csv
        valid_encoded_df.to_csv(f"tsne_{t}_{self.epoch}.csv")

        reducer = umap.UMAP()
        embedding_valid_umap = reducer.fit_transform(valid_encoded_df.values)

        xs = embedding_valid_umap[:, 0]
        ys = embedding_valid_umap[:, 1]

        df = pd.DataFrame(zip(xs, ys, y_valid_label), columns=["x", "y", "label"])

        # TODO: add additional umap plot for scatter; also, update legend name.
        sns.kdeplot(data=df, x="x", y="y", hue="label", fill=True, levels=5, alpha=0.5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP Projection of Encoded Space', fontsize=12)
        plt.savefig(f"snn_{t}_epoch_{self.epoch}_density_area.png",
                    bbox_inches="tight",
                    transparent=True)

        sns.kdeplot(data=df, x="x", y="y", hue="label", fill=False, levels=5)
        sns.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP Projection of Encoded Space', fontsize=12)
        plt.savefig(f"snn_{t}_epoch_{self.epoch}_validation_data.png",
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
