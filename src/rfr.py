import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

class RfrModel(object):
    def __init__(self, rfr):
        self.rfr = rfr
    
    def fit(self, X, y, T, v):
        return self.rfr.fit(X, y)
        
    def predict(self, X):
        return self.rfr.predict(X)

def resolve():
    ###  read the train, test and macro files
    train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
    print(train_df.shape, test_df.shape)
    
    # combine macro information with train and test
    train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
    test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
    print(train_df.shape, test_df.shape)
    
    # undersampling by magic numbers
#     trainsub = train_df[train_df.timestamp < '2015-01-01']
#     trainsub = trainsub[trainsub.product_type=="Investment"]
#     ind_1m = trainsub[trainsub.price_doc <= 1000000].index
#     ind_2m = trainsub[trainsub.price_doc == 2000000].index
#     ind_3m = trainsub[trainsub.price_doc == 3000000].index
#     train_index = set(train_df.index.copy())
#     for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
#         ind_set = set(ind)
#         ind_set_cut = ind.difference(set(ind[::gap]))
#         train_index = train_index.difference(ind_set_cut)
        
    ###  convert categorical variables into numerical variables by label encoding
    objlist = []
    for f in train_df.columns:
        if train_df[f].dtype=='object':
            objlist.append(f)       
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
            train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
            test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
            
    # year and month #
    train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
    test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month
    # year and week #
    train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
    test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear
    # year #
    train_df["year"] = train_df["timestamp"].dt.year
    test_df["year"] = test_df["timestamp"].dt.year
    # month of year #
    train_df["month_of_year"] = train_df["timestamp"].dt.month
    test_df["month_of_year"] = test_df["timestamp"].dt.month
    # week of year #
    train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
    test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear
    # day of week #
    train_df["day_of_week"] = train_df["timestamp"].dt.weekday
    test_df["day_of_week"] = test_df["timestamp"].dt.weekday

    ### We could potentially add more variables like this. But for now let us start with model building using these additional variables. Let us drop the variables which are not needed in model building.
    train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
    test_X = test_df.drop(["id", "timestamp"] , axis=1)
    # Since our metric is "RMSLE", let us use log of the target variable for model building rather than using the actual target variable.
    # train_y = np.log1p(train_df.price_doc.values)
    train_y =(train_df.price_doc.values)
    
    train_X.fillna(0, inplace=True)
    test_X.fillna(0, inplace=True)

    return train_df['id'].values, train_X.values, train_y, test_X.values