from pathlib import Path
import pandas as pd
from utils.common import SCLALER, get_folder, DIR, get_file, FILE, EXT
from utils.processing_train_test import get_matrices, get_labels
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def run_SVM(no_X: bool, fimo: bool):

    X_train, X_test = get_matrices(
        dir_type=DIR.SVM_TRAIN_TEST, scaler=SCLALER.NONE, no_X=no_X, fimo=fimo)

    y_train, y_test = get_labels(no_X=no_X, fimo=fimo)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            SVC(), tuned_parameters, cv=1
        )
        clf.fit(X_train, y_train.values.ravel())

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print(clf.cv_results_)
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
