# %%
import gc
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from utils.common import DIR, MODEL, get_folder, SCLALER, get_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from typing import List
from joblib import load

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


def create_train_test_folders(dir_type: DIR, no_X: bool, fimo: bool, recreate: bool):
    train_test_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo, recreate=recreate)

    for scaler in SCLALER:
        train_test_folder.joinpath(scaler.value).mkdir(exist_ok=True)


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


def save_matrices(matrices: List[pd.DataFrame], dir_type: DIR, scaler: SCLALER,  no_X: bool, fimo: bool, num_compnents: int = 0):
    train_test_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo)
    lda_component = ''
    if num_compnents > 0:
        lda_component = '_{}'.format(num_compnents)
    pd.DataFrame(matrices[0]).to_csv(
        train_test_folder/scaler.value/"X_train{}.csv".format(lda_component))
    pd.DataFrame(matrices[1]).to_csv(
        train_test_folder/scaler.value/"X_test{}.csv".format(lda_component))


def get_matrices(dir_type: DIR, scaler: SCLALER,  no_X: bool, fimo: bool, n_comp: int = 0):
    train_test_folder = get_folder(
        dir_type=dir_type, no_X=no_X, fimo=fimo).joinpath(scaler.value)

    if dir_type == DIR.LDA_TRAIN_TEST:
        return pd.read_csv(train_test_folder/"X_train_{}.csv".format(n_comp), index_col=0),\
            pd.read_csv(train_test_folder /
                        "X_test_{}.csv".format(n_comp), index_col=0)

    return pd.read_csv(train_test_folder/"X_train.csv", index_col=0),\
        pd.read_csv(train_test_folder/"X_test.csv", index_col=0)


def processing_SVM_matrices(matrices: List[pd.DataFrame], no_X: bool, fimo: bool):
    dir_type = DIR.SVM_TRAIN_TEST
    create_train_test_folders(
        dir_type=dir_type, no_X=no_X, fimo=fimo, recreate=True)

    for scaler_type in SCLALER:
        scaler = get_scaler(scaler_type)
        scaler.fit(matrices[0])
        save_matrices(matrices=[scaler.transform(matrix)
                      for matrix in matrices],
                      dir_type=dir_type,
                      scaler=scaler_type,
                      no_X=no_X,
                      fimo=fimo)


def processing_LDA_matrices(matrices: List[pd.DataFrame], no_X: bool, fimo: bool, topic_range: range, recreate: bool, ignore_existed: bool):
    dir_type = DIR.LDA_TRAIN_TEST
    create_train_test_folders(
        dir_type=dir_type, no_X=no_X, fimo=fimo, recreate=recreate)

    for i in topic_range:
        model_ = get_model(model_type=MODEL.LDA,
                           n_comps=i, no_X=no_X, fimo=fimo)
        lda_model: LatentDirichletAllocation = load(model_)
        new_matrices = [lda_model.transform(matrix) for matrix in matrices]
        for scaler_type in SCLALER:
            scaler = get_scaler(scaler_type)
            scaler.fit(new_matrices[0])
            save_matrices(matrices=[scaler.transform(matrix)
                                    for matrix in new_matrices],
                          dir_type=dir_type,
                          scaler=scaler_type,
                          no_X=no_X,
                          fimo=fimo,
                          num_compnents=int(model_.stem[15:]))
        gc.collect()


def processing_train_test(no_X: bool, fimo: bool, dir_type: DIR, test_size: float = 0.4, random_state=123, topic_range: range = None, recreate: bool = False, ignore_existed: bool = True):
    matrices, labels = spliting_train_test(
        no_X=no_X, fimo=fimo, test_size=test_size, random_state=random_state)
    save_labels(labels=labels, no_X=no_X, fimo=fimo)
    if dir_type == DIR.SVM_TRAIN_TEST:
        processing_SVM_matrices(matrices=matrices, no_X=no_X, fimo=fimo)
    elif dir_type == DIR.LDA_TRAIN_TEST:
        processing_LDA_matrices(
            matrices=matrices, no_X=no_X, fimo=fimo, topic_range=topic_range, recreate=recreate, ignore_existed=ignore_existed)
