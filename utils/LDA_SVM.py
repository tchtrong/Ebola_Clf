from utils.common import DIR, get_folder
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.spliting_train_test import get_train_test_set
from sklearn import svm
from joblib import load
import pandas as pd
from pathlib import Path
from typing import List


def run_SVM_LDA(no_X: bool, fimo: bool, dimens: range, use_test: bool = False,):

    X_train, X_test, y_train, y_test = get_train_test_set(
        no_X=True, fimo=True)

    scaler = MinMaxScaler()

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    gammas = ['scale', 'auto', 0.1, 1.0, 10.0]
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # kernels = ['poly']
    # gammas = [10.0, 1.0]
    # Cs = [1.0]

    model_folder = get_folder(dir_type=DIR.LDA_MODEL, no_X=no_X, fimo=fimo)
    Path("results").mkdir(exist_ok=True)

    models: List[lda] = []
    for i in dimens:
        file = model_folder.joinpath('LDA_train_only_{}'.format(i))
        lda_model: lda = load(file)
        models.append(lda_model)

    for kernel in kernels:
        for Cs_ in Cs:
            for gamma_ in gammas:
                print('Running LDA_K_{}_C_{}_G_{}.csv'.format(
                    kernel, Cs_, gamma_))
                results = []
                for idx, model in enumerate(models):
                    X_LDA_train = model.transform(X_train)
                    X_LDA_test = model.transform(X_test)
                    X_LDA_train = scaler.fit_transform(X_LDA_train)
                    X_LDA_test = scaler.transform(X_LDA_test)
                    clf = svm.SVC(kernel=kernel, C=Cs_, gamma=gamma_,
                                  class_weight='balanced', max_iter=1000000)
                    clf.fit(X_LDA_train, y_train.values.ravel())
                    results.append(
                        clf.score(X_LDA_test, y_test.values.ravel()))
                    print('\tFinished component {}'.format(idx))

                pd.Series(results).to_csv(
                    'results/LDA_K_{}_C_{}_G_{}.csv'.format(kernel, Cs_, gamma_), index=True)
                print('Finished LDA_K_{}_C_{}_G_{}.csv'.format(
                    kernel, Cs_, gamma_))

                if kernel == 'linear':
                    break
