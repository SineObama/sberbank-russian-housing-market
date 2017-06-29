import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
import xgboost as xgb

class XgbModel(object):
    def __init__(self, params, num_boost_rounds):
        self.params = params
        self.num_boost_rounds = num_boost_rounds
        
    def copy(self):
        return XgbModel(self.params, self.num_boost_rounds)
    
    def fit(self, X, y, T, v):
        xgtrain = xgb.DMatrix(X, y)
        xgval = xgb.DMatrix(T, v)
        watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
        self.model = xgb.train(self.params, xgtrain, self.num_boost_rounds, watchlist, early_stopping_rounds=20, verbose_eval=50)
        
    def predict(self, X):
        xgtest = xgb.DMatrix(X)
        return self.model.predict(xgtest)

class Stacking(object):
    def __init__(self, n_folds, base_models, data_resolver, feval):
        '''
        data_resolver.next():id, X_train, y_train, X_test (np array)
        feval(predictions, targets):value
        '''
        self.n_folds = n_folds
        self.base_models = base_models
        self.data_resolver = data_resolver
        self.feval = feval

    def fit(self, id_all, num_test):

        kf = model_selection.KFold(n_splits=self.n_folds, shuffle=True, random_state=2016)

        S_train = np.zeros((len(id_all), len(self.base_models)))
        S_test = np.zeros((num_test, len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            id, X, y, T = self.data_resolver.next()
            S_test_i = np.zeros((T.shape[0], self.n_folds))
            folds = kf.split(range(len(y)))

            S_train_i = np.zeros(len(y))
            for j, (train_idx, test_idx) in enumerate(folds):
                print "model",i,"fold",j
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                clf.fit(X_train, y_train, X_holdout, y_holdout)
                y_pred = clf.predict(X_holdout)[:]
                print "train:",self.feval(clf.predict(X_train)[:], y_train),"val:",self.feval(y_pred, y_holdout)
                S_train_i[test_idx] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_train[:, i] = pd.DataFrame(id_all, columns=['id']).merge(pd.DataFrame({'id':id, 'y':S_train_i}), on='id', how='left')['y'].values

            S_test[:, i] = S_test_i.mean(1)

        return (S_train, S_test)