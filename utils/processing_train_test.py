# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.common import DIR, get_folder, SCLALER
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from typing import List

# %%


class NothingScaler():
    def fit(self, X):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X


def get_scaler(scaler: SCLALER):
    if scaler == SCLALER.MaxAbsScaler:
        return MaxAbsScaler()
    elif scaler == SCLALER.MinMaxScaler:
        return MinMaxScaler()
    elif scaler == SCLALER.RobustScaler:
        return RobustScaler()
    elif scaler == SCLALER.StandardScaler:
        return StandardScaler()
    elif scaler == SCLALER.NONE:
        return NothingScaler()


def spliting_train_test(no_X: bool, fimo: bool, test_size: float = 0.4, random_state=123):
    csv_folder = get_folder(DIR.CSV, no_X=no_X, fimo=fimo)
    list_csv = list(csv_folder.glob("*.csv"))
    list_csv.sort()

    lst = [[], [], [], []]
    for csv in list_csv:
        data = pd.read_csv(csv, index_col=0)
        train_test_set = train_test_split(
            data.drop('Label', axis=1), data['Label'], test_size=test_size, random_state=random_state)
        for idx, set_ in enumerate(train_test_set):
            lst[idx].append(set_)

    lst = [pd.concat(set_) for set_ in lst]

    return lst[:2], lst[2:]


def create_train_test_folders(dir_type: DIR, no_X: bool, fimo: bool):
    train_test_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo, recreate=True)

    for scaler in SCLALER:
        train_test_folder.joinpath(scaler.value).mkdir()


def save_labels(labels: List[pd.Series], no_X: bool, fimo: bool):
    labels_folder = get_folder(
        dir_type=DIR.LABELS, no_X=no_X, fimo=fimo, recreate=True)

    labels[0].to_csv(labels_folder/"y_train.csv")
    labels[1].to_csv(labels_folder/"y_test.csv")


def get_labels(no_X: bool, fimo: bool):
    labels_folder = get_folder(
        dir_type=DIR.LABELS, no_X=no_X, fimo=fimo)
    return pd.read_csv(labels_folder/"y_train.csv", index_col=0),\
        pd.read_csv(labels_folder/"y_test.csv", index_col=0)


def save_matrices(matrices: List[pd.DataFrame], dir_type: DIR, scaler: SCLALER,  no_X: bool, fimo: bool):
    matrix_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo)

    pd.DataFrame(matrices[0]).to_csv(matrix_folder/scaler.value/"X_train.csv")
    pd.DataFrame(matrices[1]).to_csv(matrix_folder/scaler.value/"X_test.csv")


def get_matrices(dir_type: DIR, scaler: SCLALER,  no_X: bool, fimo: bool):
    matrix_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo).joinpath(scaler.value)

    return pd.read_csv(matrix_folder/"X_train.csv", index_col=0),\
        pd.read_csv(matrix_folder/"X_test.csv", index_col=0)


def processing_SVM_matrices(matrices: List[pd.DataFrame], no_X: bool, fimo: bool):
    dir_type = DIR.SVM_TRAIN_TEST
    create_train_test_folders(
        dir_type=dir_type, no_X=no_X, fimo=fimo)

    for scaler_type in SCLALER:
        scaler = get_scaler(scaler_type)
        scaler.fit(matrices[0])
        print(matrices[0])
        save_matrices(matrices=[scaler.transform(matrix)
                      for matrix in matrices],
                      dir_type=dir_type,
                      scaler=scaler_type,
                      no_X=no_X,
                      fimo=fimo)


def processing_train_test(no_X: bool, fimo: bool, test_size: float = 0.4, random_state=123):
    matrices, labels = spliting_train_test(
        no_X=no_X, fimo=fimo, test_size=test_size, random_state=random_state)

    save_labels(labels=labels, no_X=no_X, fimo=fimo)
    processing_SVM_matrices(matrices=matrices, no_X=no_X, fimo=fimo)

