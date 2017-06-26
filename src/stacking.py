import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
class Stacking(object):
    def __init__(self, n_folds, base_models, data_resolver, feval):
        '''
        data_resolver.next():X_train, y_train, X_test (np array)
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
            print "model",i
            folds = kf.split(range(len(y)))

            S_train_i = np.zeros(len(y))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                print "train:",self.feval(clf.predict(X_train)[:], y_train),"val:",self.feval(y_pred, y_holdout)
                S_train_i[test_idx] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_train[:, i] = pd.DataFrame(id_all, columns=['id']).merge(pd.DataFrame({'id':id, 'y':S_train_i}), on='id', how='left')['y'].values

            S_test[:, i] = S_test_i.mean(1)

        return (S_train, S_test)