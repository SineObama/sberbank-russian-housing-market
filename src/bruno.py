#encoding: utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
'''
class Resolver(object):
    def __init__(self):
        self.__have = False

    def clear(self):
        self.__have = False
        del self.X_train
        del self.y_train
        del self.X_test

    def get(self):
        if (self.__have == True):
            return self.X_train, self.y_train, self.X_test'''
def resolve():
    df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
    df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

    df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

    mult = 0.969
    id_train = df_train['id'].values
    y_train = df_train['price_doc'].values * mult + 10
    id_test = df_test['id']

    df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
    df_test.drop(['id'], axis=1, inplace=True)

    num_train = len(df_train)
    df_all = pd.concat([df_train, df_test])
    # Next line just adds a lot of NA columns (becuase "join" only works on indexes)
    # but somewhow it seems to affect the result
    df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
    #print(df_all.shape)

    # Add month-year
    month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    df_all['month'] = df_all.timestamp.dt.month
    df_all['dow'] = df_all.timestamp.dt.dayofweek

    # Other feature engineering
    df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
    df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
    ''' 未知无效的插入代码？
    train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
    test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]

    def add_time_features(col):
       col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
       train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

       col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
       train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

    add_time_features('building_name')
    add_time_features('sub_area')

    def add_time_features(col):
       col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])
       test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

       col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])
       test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

    add_time_features('building_name')
    add_time_features('sub_area')
    '''

    # Remove timestamp column (may overfit the model in train)
    df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


    factorize = lambda t: pd.factorize(t[1])[0]

    df_obj = df_all.select_dtypes(include=['object'])

    X_all = np.c_[
        df_all.select_dtypes(exclude=['object']).values,
        np.array(list(map(factorize, df_obj.iteritems()))).T
    ]
    #print(X_all.shape)

    X_train = X_all[:num_train]
    X_test = X_all[num_train:]


    # Deal with categorical values
    df_numeric = df_all.select_dtypes(exclude=['object'])
    df_obj = df_all.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_values = pd.concat([df_numeric, df_obj], axis=1)
    
    # Convert to numpy values
    X_all = df_values.values
    #print(X_all.shape)

    X_train = X_all[:num_train]
    X_test = X_all[num_train:]

    return id_train, X_train, y_train, X_test
        
'''self.X_train = X_train
                        self.y_train = y_train
                        self.X_test = X_test
                        self.__have = True
                        return self.X_train, self.y_train, self.X_test'''