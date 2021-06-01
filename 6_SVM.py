# %%
from sklearn import svm
from utils.spliting_train_test import get_train_test_set
from pathlib import Path
import pandas as pd


def run_SVM(is_fimo: bool, is_cleaned: bool):
    result_folder = Path("results")
    result_folder.mkdir(exist_ok=True)

    file_result = "SVM.csv"

    X_train, X_test, y_train, y_test = get_train_test_set(is_fimo, is_cleaned)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    gammas = ['scale', 'auto', 0.1, 1.0, 1.0, 10.0]
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    results = []
    for kernel in kernels:
        for Cs_ in Cs:
            for gamma_ in gammas:
                clf = svm.SVC(kernel=kernel, C=Cs_, gamma=gamma_)
                clf.fit(X_train, y_train.values.ravel())
                score = clf.score(X_test, y_test.values.ravel())
                results.append({'Kernel': kernel, "C": Cs_,
                               "Gamma": gamma_, "Result": score})
    pd.DataFrame(results).to_csv(result_folder/file_result, index=True)


# %%
run_SVM(is_fimo=True, is_cleaned=False)
