from pathlib import Path
from utils.common import SCLALER, DIR, get_folder
from utils.processing_train_test import get_matrices, get_labels
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def create_result_folder(dir_type: DIR, no_X: bool, fimo: bool, recreate: bool = False):
    result_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo, recreate=recreate)
    for scaler in SCLALER:
        result_folder.joinpath(scaler.value).mkdir(exist_ok=True)


def get_param_grid(kernel: str):
    if kernel == 'linear':
        C_range = np.logspace(start=-3, stop=3, num=7, base=10)
        param_grid = dict(C=C_range)
        return param_grid
    elif kernel == 'rbf':
        C_range = np.logspace(start=-3, stop=9, num=13, base=10)
        gamma_range = np.logspace(start=-9, stop=3, num=13, base=10)
        param_grid = dict(gamma=gamma_range, C=C_range)
        return param_grid


def run_SVM_only(no_X: bool, fimo: bool, kernel: str):

    create_result_folder(DIR.SVM_RESULTS, no_X=no_X, fimo=fimo)
    result_folder = get_folder(DIR.SVM_RESULTS, no_X=no_X, fimo=fimo)

    y_train, y_test = get_labels(no_X=no_X, fimo=fimo)
    y = pd.concat([y_train, y_test]).values.ravel()

    for scaler_type in SCLALER:
        X_train, X_test = get_matrices(
            dir_type=DIR.SVM_TRAIN_TEST, scaler=scaler_type, no_X=no_X, fimo=fimo)
        X = pd.concat([X_train, X_test])

        Path(result_folder/scaler_type.value).joinpath(kernel).mkdir(exist_ok=True)
        param_grid = get_param_grid(kernel)

        cv = StratifiedKFold(n_splits=3)
        grid = GridSearchCV(SVC(kernel=kernel, max_iter=10000000),
                            param_grid=param_grid, cv=cv, n_jobs=-1)
        grid.fit(X, y)

        scores = None
        if kernel == 'rbf':
            scores = grid.cv_results_['mean_test_score'].reshape(
                len(param_grid['C']), len(param_grid['gamma']))
            scores = pd.DataFrame(
                scores, index=param_grid['C'], columns=param_grid['gamma'])
        elif kernel == 'linear':
            scores = grid.cv_results_['mean_test_score'].reshape(
                len(param_grid['C']), 1)
            scores = pd.DataFrame(scores, index=param_grid['C'])
        scores.to_csv(result_folder/scaler_type.value /
                      kernel/'mean_test_score.csv')

        with open(result_folder/scaler_type.value/kernel/'best_score_best_params.txt', 'w') as file:
            file.write('{}\t\t\t{}'.format(
                grid.best_params_, grid.best_score_))


def run_SVM_LDA(no_X: bool, fimo: bool, topic_range: range, kernel: str):

    create_result_folder(DIR.LDA_RESULTS, no_X=no_X, fimo=fimo)
    result_folder = get_folder(DIR.LDA_RESULTS, no_X=no_X, fimo=fimo)

    y_train, y_test = get_labels(no_X=no_X, fimo=fimo)
    y = pd.concat([y_train, y_test]).values.ravel()

    for scaler_type in SCLALER:
        if scaler_type == SCLALER.RobustScaler:
            continue
        Path(result_folder/scaler_type.value).joinpath(kernel).mkdir(exist_ok=True)
        result_to_print = ''
        for i in topic_range:
            X_train, X_test = get_matrices(
                dir_type=DIR.LDA_TRAIN_TEST, scaler=scaler_type, no_X=no_X, fimo=fimo, n_comp=i)
            X = pd.concat([X_train, X_test])
            Path(result_folder/scaler_type.value).joinpath(kernel).mkdir(exist_ok=True)
            param_grid = get_param_grid(kernel)

            cv = StratifiedKFold(n_splits=3)
            grid = GridSearchCV(SVC(kernel=kernel, max_iter=10000000),
                                param_grid=param_grid, cv=cv, n_jobs=-1)
            grid.fit(X, y)

            scores = None
            if kernel == 'rbf':
                scores = grid.cv_results_['mean_test_score'].reshape(
                    len(param_grid['C']), len(param_grid['gamma']))
                scores = pd.DataFrame(
                    scores, index=param_grid['C'], columns=param_grid['gamma'])
            elif kernel == 'linear':
                scores = grid.cv_results_['mean_test_score'].reshape(
                    len(param_grid['C']), 1)
                scores = pd.DataFrame(scores, index=param_grid['C'])
            scores.to_csv(result_folder/scaler_type.value /
                          kernel/'mean_test_score_{}.csv'.format(i))
            result_to_print += str(i) + '\t' + str(grid.best_params_) + \
                '\t' + str(grid.best_score_) + '\n'

        with open(result_folder/scaler_type.value/kernel/'best_score_best_params.txt', 'w') as file:
            file.write(result_to_print)
