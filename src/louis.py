import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

def resolve():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    id_test = test.id

    mult = .969

    y_train = train["price_doc"] * mult + 10
    id_train = train["id"].values
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))

    return id_train, x_train.values, y_train.values, x_test.values