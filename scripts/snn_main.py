"""Main module."""
import argparse
import copy
import logging
import os

import fastai
import numpy as np
import torch
from fastai.callback.tracker import SaveModelCallback
from fastai.data.core import DataLoaders, range_of
from fastai.layers import LinBnDrop
from fastai.learner import Learner
from fastai.metrics import F1Score, Precision, Recall, accuracy
from fastai.tabular.all import (Categorify, CategoryBlock, FillMissing,
                                Normalize, RandomSplitter, TabDataLoader,
                                TabularPandas, tabular_config, tabular_learner)
from fastai.torch_basics import params
from loaderbot.big_query import query_table_and_cache
from mobius.callbacks import TSNECallback
from mobius.datasets import TabularSiameseDataset, write_jsonl
from mobius.losses import ContrastiveLoss
from mobius.models import TabularSiameseModel
from mobius.utils import emb_sz_rule
from sklearn.model_selection import train_test_split


def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]


def contrastive_loss_func(out, targ):
    return ContrastiveLoss(margin=1.0)(out, targ.long())


# TODO: put this on gpu
def snn(params: dict):
    device = params["device"]
    fastai.device = torch.device(device)

    # open and read sql query
    fd = open('snn_main.sql', 'r')
    sql = fd.read()
    fd.close()

    df = query_table_and_cache(sql=sql)

    if params["sample"] < 1.:
        df, _ = train_test_split(
            df,
            test_size=(1 - params["sample"]),
            stratify=df["label"])

    logging.info(f"df rows, cols: {df.shape}")

    exclude_vars = ["label", "id", "investorId", "createdAt", "investorId_1", "investorId_2",
                    "investorLevel", "investorLevel_1", "status", "windfall_id", "windfall_id_1",
                    "candidate_id", "minInvestmentDate", "maxInvestmentDate", "confidence",
                    "closed", "countInvestmentDate", "amount", "sumAmount"]

    y_names = ["label"]
    y_block = CategoryBlock()

    cat_names = [x for x in df.select_dtypes(
        exclude=['int', 'float']).columns if x != y_names]
    cat_names = [x for x in cat_names if x not in exclude_vars]

    # calc embedding sizes for each categorical feature
    emb_szs = {k: emb_sz_rule(len(df[k].unique())) for k in cat_names}
    emb_szs

    cont_names = [x for x in df.select_dtypes(
        [np.number]).columns if x != y_names]
    cont_names = [x for x in cont_names if x not in exclude_vars]
    cont_names

    procs = [FillMissing, Categorify, Normalize]

    # train/test split
    splits = RandomSplitter(valid_pct=0.20)(range_of(df))

    tabular_pandas = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=y_names,
        y_block=y_block,
        splits=splits,
        device=device)

    trn_dl = TabDataLoader(
        tabular_pandas.train,
        bs=params["tabular_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4)

    val_dl = TabDataLoader(
        tabular_pandas.valid,
        bs=params["tabular_batch_size"],
        num_workers=4)

    dls = DataLoaders(trn_dl, val_dl)
    if device == "cuda":  # put tabular dataloader on gpu
        dls = dls.cuda()

    # load the tabular_pandas data through the tabular_learner
    layers = [params["tabular_layer_1_neurons"],
              params["tabular_layer_2_neurons"],
              params["tabular_layer_3_neurons"]]

    # tabular learner configuration
    config = tabular_config(ps=[params["tabular_layer_dropout"]] * len(layers),
                            embed_p=params["tabular_embed_dropout"])

    learn = tabular_learner(
        dls,
        layers=layers,
        emb_szs=emb_szs,
        config=config,
        metrics=[accuracy,
                 Precision(average='macro'),
                 Recall(average='macro'),
                 F1Score(average='macro')])

    learn.fit_one_cycle(n_epoch=params["tabular_n_epoch"])

    if not os.path.exists('data'):
        os.makedirs('data')

    write_jsonl(
        tabular_pandas.train.to.items[0].items, "data/train_data.jsonl")
    write_jsonl(
        tabular_pandas.valid.to.items[0].items, "data/valid_data.jsonl")

    # write SNN training labels to `data/`
    tabular_pandas.train.y.to_csv("data/train_labels.csv", index=True)
    tabular_pandas.valid.y.to_csv("data/valid_labels.csv", index=True)

    train_ds = TabularSiameseDataset(
        csv_file="data/train_labels.csv",
        jsonl_file="data/train_data.jsonl",
        tabular_learner=learn)

    valid_ds = TabularSiameseDataset(
        csv_file="data/valid_labels.csv",
        jsonl_file="data/valid_data.jsonl",
        tabular_learner=learn)

    dls = DataLoaders.from_dsets(
        train_ds,
        valid_ds,
        bs=params["snn_batch_size"],
        num_workers=params["snn_n_workers"],
        device=device)

    encoder = copy.copy(learn)
    encoder.model.layers = learn.model.layers[:-1]
    encoder_model = encoder.model

    head = LinBnDrop(n_in=layers[-1]*2,
                     n_out=params["snn_n_out"],
                     bn=True,
                     act=None)

    model = TabularSiameseModel(encoder_model, head)
    if device == "cuda":  # put snn on gpu
        model = model.cuda()

    # add callback for best validation epoch
    siamese_learner = Learner(dls,
                              model,
                              model_dir=params["model_dir"],
                              loss_func=contrastive_loss_func,
                              splitter=siamese_splitter,
                              cbs=[TSNECallback, SaveModelCallback])

    siamese_learner.unfreeze()
    siamese_learner.fit(n_epoch=params["snn_n_epoch"], lr=params["snn_lr"])


def main(args):
    parser = argparse.ArgumentParser(description="tabular_neural_net")
    parser.add_argument("--tabular_lr", required=True, help="...")
    parser.add_argument("--tabular_n_epoch", required=True, help="...")
    parser.add_argument("--tabular_n_workers", required=True, help="...")
    parser.add_argument("--tabular_batch_size", required=True, help="...")
    parser.add_argument("--tabular_layer_dropout", required=True, help="...")
    parser.add_argument("--tabular_embed_dropout", required=True, help="...")
    parser.add_argument("--tabular_layer_1_neurons", required=True, help="...")
    parser.add_argument("--tabular_layer_2_neurons", required=True, help="...")
    parser.add_argument("--tabular_layer_3_neurons", required=True, help="...")
    parser.add_argument("--snn_lr", required=True, help="...")
    parser.add_argument("--snn_n_out", required=True, help="...")
    parser.add_argument("--snn_margin", required=True, help="...")
    parser.add_argument("--snn_n_epoch", required=True, help="...")
    parser.add_argument("--snn_n_workers", required=True, help="...")
    parser.add_argument("--snn_batch_size", required=True, help="...")
    parser.add_argument("--model_dir", required=True, help="...")
    parser.add_argument("--device", required=True, help="...")
    parser.add_argument("--sample", required=True, help="...")
    args = parser.parse_args(args[1:])

    # unpack cli args into params dict
    params = {i: eval(args.__getattribute__(i))
              for i in args.__dir__() if i[0] != '_'}

    snn(params)
    logging.info("Job complete!")


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
