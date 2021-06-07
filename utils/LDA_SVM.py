from utils.common import DIR, get_folder
from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from sklearn import svm
from joblib import load
import pandas as pd
from pathlib import Path
import gc


def run_SVM_LDA(no_X: bool, fimo: bool, dimens: range, use_test: bool = False,):

    X_train, X_test, y_train, y_test = get_train_test_set(
        no_X=True, fimo=True)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    gammas = ['scale', 'auto', 0.1, 1.0, 10.0]
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # kernels = ['linear']
    # gammas = ['scale']
    # Cs = [100.0]
    model_folder = get_folder(dir_type=DIR.LDA_MODEL, no_X=no_X, fimo=fimo)
    Path("results").mkdir(exist_ok=True)

    for kernel in kernels:
        for Cs_ in Cs:
            for gamma_ in gammas:
                if kernel == 'linear' and Cs_ >= 100.0 or kernel == 'poly' and type(gamma_) == float and gamma_ >= 10.0:
                    break
                results = []
                for i in dimens:
                    file = model_folder.joinpath('LDA_train_only_{}'.format(i))
                    lda_model: lda = load(file)
                    X_LDA_train = lda_model.transform(X_train)
                    X_LDA_test = lda_model.transform(X_test)
                    clf = svm.SVC(kernel=kernel, C=Cs_, gamma=gamma_,
                                  class_weight='balanced')
                    clf.fit(X_LDA_train, y_train.values.ravel())
                    results.append(
                        clf.score(X_LDA_test, y_test.values.ravel()))
                    del lda_model
                    del clf
                    del X_LDA_test
                    del X_LDA_train
                    gc.collect()

                pd.Series(results).to_csv(
                    'results/LDA_K_{}_C_{}_G_{}.csv'.format(kernel, Cs_, gamma_), index=True)
                print('Finished LDA_K_{}_C_{}_G_{}.csv'.format(
                    kernel, Cs_, gamma_))

                del results
                gc.collect()

                if kernel == 'linear':
                    break
