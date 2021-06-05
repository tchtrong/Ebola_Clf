# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from utils.common import DIR, get_folder


def get_name_train_test_set(no_X: bool, fimo: bool) -> list[str]:
    lst = ["X_train", "X_test", "y_train", "y_test"]
    for idx, _ in enumerate(lst):
        if fimo:
            lst[idx] += "_fimo"
        if no_X:
            lst[idx] += "_no_X"
        lst[idx] += ".csv"
    return lst


def write_train_test_set_to_csv(lst: list, list_file: list):
    train_test_set_folder = Path("train_test")
    train_test_set_folder.mkdir(exist_ok=True)
    for idx, file in enumerate(list_file):
        pd.concat(lst[idx]).to_csv(train_test_set_folder / file)


def spliting_train_test(no_X: bool, fimo: bool, test_size: float = 0.4, random_state=123):
    csv_folder = get_folder(DIR.CSV, no_X=no_X, fimo=fimo)
    list_csv = list(csv_folder.glob("*.csv"))
    lst = [[], [], [], []]
    for csv in list_csv:
        data = pd.read_csv(csv, index_col=0)
        train_test_set = train_test_split(
            data.drop('Label', axis=1), data['Label'], test_size=test_size, random_state=random_state)
        for idx, set_ in enumerate(train_test_set):
            lst[idx].append(set_)
    train_test_names = get_name_train_test_set(no_X=no_X, fimo=fimo)
    write_train_test_set_to_csv(lst, train_test_names)


def get_train_test_set(no_X: bool, fimo: bool) -> list[pd.DataFrame]:
    files = get_name_train_test_set(no_X=no_X, fimo=fimo)
    train_test_set = []
    train_test_set_folder = Path("train_test")
    for file in files:
        train_test_set.append(pd.read_csv(
            train_test_set_folder / file, index_col=0))
    return train_test_set
