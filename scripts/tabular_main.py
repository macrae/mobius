"""Main module."""
import argparse
import logging
import warnings

import loaderbot.big_query as bq
import numpy as np
import pandas as pd
from fastai.callback.tracker import SaveModelCallback
from fastai.data.core import DataLoaders, range_of
from fastai.metrics import F1Score, Precision, Recall, accuracy
from fastai.tabular.all import (Categorify, CategoryBlock, FillMissing,
                                Normalize, RandomSplitter, TabDataLoader,
                                TabularPandas, tabular_config, tabular_learner)
from fastai.torch_basics import params
from google.cloud import bigquery, storage
from jinjasql import JinjaSql
from mobius.utils import emb_sz_rule
from roc_it.ml.binary_classification import BinaryClassification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from zoolander.data_processing import run_tree_based_feature_selection

import config.label_alternative_investment as config_label_alternative_investment
import config.label_education_donor as config_label_education_donor
import config.label_environment_donor as config_label_environment_donor
import config.label_healthcare_donor as config_label_healthcare_donor
import config.label_lux_goods as config_label_lux_goods
import config.label_lux_travel as config_label_lux_travel
from fastai.callback.training import ShortEpochCallback

# GCP Clients
bigquery_client = bigquery.Client(project="tranquil-garage-139216")
storage_client = storage.Client(project="tranquil-garage-139216")

warnings.filterwarnings('ignore')
pd.options.display.max_rows = 50
pd.options.display.max_columns = 999

def data_prep(config, load_from_bq=False, print_sql=False):
    # load sql template
    fd = open('./train.sql', 'r')
    sql_template = fd.read()
    fd.close()
    # populate sql template with params
    j = JinjaSql(param_style='pyformat')
    query, query_params = j.prepare_query(sql_template, config.params)
    sql = query % query_params
    if print_sql:
        print(f"sql query:\n{sql}")  
    label_name = config.params["target"]    
    if load_from_bq:
        # query training data
        raw_data = bq.query_table(
            sql=sql,
            client=bigquery_client
        )
        raw_data.to_csv(f"./data/df_{label_name}.csv")
    else:
        raw_data = pd.read_csv(f"./data/df_{label_name}.csv")
    # exclude variables
    exclude_vars = list(set(raw_data.columns) - set(config.include_vars))
    return raw_data, exclude_vars


def preprocess_data(data_train, data_val, exclude_vars):
    # prep
    df_train = data_train.drop(columns = ['label'] + exclude_vars)
    df_val = data_val.drop(columns = ['label'] +  exclude_vars)
    features = list(df_train.columns)
    # impute
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(df_train)
    df_train = imputer.transform(df_train)
    df_val = imputer.transform(df_val)
    # normalize
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(df_train)
    df_train_ = scaler.transform(df_train)
    df_val_ = scaler.transform(df_val)
    # prep dfs
    df_train = pd.DataFrame(df_train_, columns=features)
    df_train.loc[:, "label"] = data_train["label"].values
    df_val = pd.DataFrame(df_val_, columns=features)
    df_val.loc[:, "label"] = data_val["label"].values
    print('Training data shape: ', df_train.shape)
    print('Validation data shape: ', df_val.shape)
    return df_train, df_val

def snn(params: dict):
    LABEL_NAME = params["label_name"]
    if LABEL_NAME == "label_alternative_investment":
        config = config_label_alternative_investment
    elif LABEL_NAME == "label_education_donor":
        config = config_label_education_donor
    elif LABEL_NAME == "label_environment_donor":
        config = config_label_environment_donor
    elif LABEL_NAME == "label_healthcare_donor":
        config = config_label_healthcare_donor
    elif LABEL_NAME == "label_lux_goods":
        config = config_label_lux_goods
    elif LABEL_NAME == "label_lux_travel":
        config = config_label_lux_travel
    config.params

    raw_data, exclude_vars = data_prep(config)

    df = raw_data.sample(1_000)
    df_train_raw, df_val_raw = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"])

    df_train, df_val = preprocess_data(df_train_raw, df_val_raw, exclude_vars)

    X_train = run_tree_based_feature_selection(
        X_train = df_train.drop(columns=["label"]),
        y_train = df_train.label,
        model = ExtraTreesClassifier(n_estimators=100),
        max_features= None,
        threshold = None,
    ) 

    df_train_reduced = X_train.copy()
    df_train_reduced.loc[:, "label"] = df_train["label"].values
    df_val_reduced = df_val.copy()[X_train.columns]
    df_val_reduced.loc[:, "label"] = df_val["label"].values
    df_train_reduced.head()

    # fit a model
    clf =  RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -2)
    clf.fit(df_train_reduced.drop(columns=["label"]), df_train_reduced["label"])
    # Extract feature importances
    feature_importance_values = clf.feature_importances_
    feature_importances = pd.DataFrame({
        'feature': df_train_reduced.drop(columns=["label"]).columns, 
        'importance': feature_importance_values
    })
    feature_importances

    y_train_scores = pd.Series(clf.predict_proba(df_train_reduced.drop(columns=["label"]))[:, 1])
    y_val_scores = pd.Series(clf.predict_proba(df_val_reduced.drop(columns=["label"]))[:, 1])
    BinaryClassification(df_train_reduced.label, y_train_scores).save_artifacts(f"./artifacts", "train")
    BinaryClassification(df_val_reduced.label, y_val_scores).save_artifacts(f"./artifacts", "test")

    # put the data back together
    df = pd.concat([df_train_reduced, df_val_reduced])
    df = df.sample(frac=1)
    print('Data shape: ', df.shape)
    
    df = raw_data[df.columns].copy()
    df.dropna(subset=["metroRank"], inplace=True)
    print('Data shape: ', df.shape)

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
        device="cpu")

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

    # score MV leads
    sql = """
    WITH
      audience AS (
        SELECT *
        FROM `tranquil-garage-139216.people.audience_latest` a
        WHERE id IS NOT NULL),

      matches AS (
        SELECT DISTINCT
            account_id,
            windfall_id ,
            candidate_id,
            confidence
        FROM `portal.match`
        WHERE account_id = 753),

        leads AS (
            SELECT DISTINCT
              id AS investorId,
              m.windfall_id AS windfall_id,
              PARSE_TIMESTAMP("%Y-%m-%d %H:%M:%S", l.created_at) AS createdAt,
            FROM
            `portal_sync.account_753_dataset_0_file_1251551_mv_windfall_data_2021_05_12_csv` l
            LEFT JOIN matches m ON l.id = m.candidate_id
          )

    SELECT DISTINCT
      l.investorId,
      l.windfall_id AS id,
      a.* EXCEPT(id),
      dbusa.* EXCEPT(id)
      FROM leads l
      LEFT JOIN audience a ON l.windfall_id = a.id
      LEFT JOIN matches m ON l.windfall_id = m.windfall_id
      LEFT JOIN people.audience_dbusa_features dbusa using(id)
      WHERE m.candidate_id IS NOT NULL
    """

    mv_leads = bq.query_table_and_cache(sql=sql)
    mv_cols = [col for col in df.columns.values if col != "label"]
    mv_leads = mv_leads[mv_cols]
    mv_leads.dropna(subset=["metroRank"], inplace=True)
    # mv_leads.dropna(subset=["minHouseholdAge"], inplace=True)

    to = learn.dls.train_ds.new(mv_leads)
    to.process()

    # create the tabular dataloader
    # dl = TabDataLoader(to)
    dl = learn.dls.test_dl(to, bs=64)

    logging.info("Making predictions ...")
    preds, *_ = learn.get_preds(dl=dl)
    labels = np.argmax(preds, 1)

    # combine windfall_ids, with model label and scores
    results = pd.concat([
        df["id"].reset_index(drop=True),
        pd.Series(labels),
        pd.Series(preds[:, 1])
    ], axis=1
    )
    results.columns = ["id", "label", "score"]

    # write scored results to file
    results.to_csv(f"mv_scores_{params['label_name']}.csv", index=False)


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
    parser.add_argument("--label_name", required=True, help="...")
    args = parser.parse_args(args[1:])

    # unpack cli args into params dict
    params = {i: eval(args.__getattribute__(i))
              for i in args.__dir__() if i[0] != '_'}

    snn(params)
    logging.info("Job complete!")


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
