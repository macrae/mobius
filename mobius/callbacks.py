import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from fastai.callback.core import Callback
from joblib import Parallel, delayed
from seaborn.distributions import kdeplot
from sklearn.manifold import TSNE

from mobius.charts import plot_3d

WINDFALL_PRIMARY_COLORS = [
    "#1fa5c1",  # windfall blue
    "#cccccc",  # grey
    "#b6d7a8",  # green
    "#fce5cd"  # yellow
]

class TSNECallback(Callback):
    def after_validate(self):
        # # half of the time run the callback
        # if random.random() < 0.5:
        #     return
        plt.clf()
        plt.figure(figsize=(12, 8), dpi=200)
        t = int(time.time())
        valid_encoded = list()

        # if self.epoch % 2 == 0:

        # TODO: sample validation data...
        for i in range(len(self.dls.valid.dataset.labels)):
            x, y = self.dls.valid_ds.__getitem__(i)

            # rehsape into mini-batch size 1
            p1 = [t.unsqueeze(0) for t in x[0]]

            # encode the household into output embedding space
            p_encode = self.model.encode(p1)
            valid_encoded.append(p_encode)

        y_valid_label = self.dls.valid.dataset.labels["label"]
        valid_encoded_df = pd.DataFrame(torch.stack(valid_encoded).squeeze())

        # write encoded space to csv
        valid_encoded_df.to_csv(f"tsne_{t}_{self.epoch}.csv")

        # TODO: look at joblib.parallel => x do 1 thing... process pool
        reducer = umap.UMAP() 
        # embedding_valid_umap = Parallel(n_jobs=1)(delayed(reducer.fit_transform)()(valid_encoded_df.values))
        embedding_valid_umap = reducer.fit_transform(valid_encoded_df.values)

        xs = embedding_valid_umap[:, 0]
        ys = embedding_valid_umap[:, 1]

        kde_plots = dict()
        for label in set(y_valid_label):
            kde_plots[label] = dict()
            kde_plots[label]["x"] = [x for x, l in zip(xs, y_valid_label) if l == label]
            kde_plots[label]["y"] = [y for y, l in zip(ys, y_valid_label) if l == label]

        # TODO: add additional umap plot for scatter; also, update legend name.
        for i, label in enumerate(kde_plots):
            x = kde_plots[label]["x"]
            y = kde_plots[label]["y"]
            sns.scatterplot(x, y, color=WINDFALL_PRIMARY_COLORS[i], alpha=0.5, s=12)
            sns.kdeplot(x, y, color=WINDFALL_PRIMARY_COLORS[i], levels=5, shade=False, shade_lowest=True)

        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP Projection of Encoded Space', fontsize=12)
        plt.savefig(f"snn_{t}_epoch_{self.epoch}_density_area.png",
                    bbox_inches="tight",
                    transparent=True)
