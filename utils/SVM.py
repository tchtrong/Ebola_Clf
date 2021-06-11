from utils.common import SCLALER, DIR, get_dataset
from utils.processing_train_test import get_matrices, get_labels
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def run_SVM(no_X: bool, fimo: bool):

    X_train, X_test = get_matrices(
        dir_type=DIR.SVM_TRAIN_TEST, scaler=SCLALER.NONE, no_X=no_X, fimo=fimo)

    y_train, y_test = get_labels(no_X=no_X, fimo=fimo)

    dataset = get_dataset(no_X=no_X, fimo=fimo)
    X = dataset.drop('Label', axis=1)
    y = dataset['Label']

    X = pd.concat([X_train, X_test]).sort_index()
    y = pd.concat([y_train, y_test]).sort_index().values.ravel()

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    kernels = ['rbf', 'linear']
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernels)
    cv = StratifiedKFold(n_splits=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=-1)
    grid.fit(X, y)
    # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
    #                                                      len(gamma_range))

    print(grid.best_params_, grid.best_score_)