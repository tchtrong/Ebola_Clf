# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


def get_csv_folder(is_fimo: bool, is_cleaned: bool) -> Path:
    csv_folder = "csv"
    if is_fimo:
        csv_folder += "_fimo"
    if is_cleaned:
        csv_folder += "_cleaned"
    csv_folder += "_no_X"
    csv_folder = Path(csv_folder)
    return csv_folder


def get_name_train_test_set(is_fimo: bool, is_cleaned: bool) -> list:
    lst = ["X_train", "X_test", "y_train", "y_test"]
    for idx, _ in enumerate(lst):
        if is_fimo:
            lst[idx] += "_fimo"
        if is_cleaned:
            lst[idx] += "_cleaned"
        lst[idx] += "_no_X.csv"
    return lst


def write_train_test_set_to_csv(lst: list, list_file: list):
    train_test_set_folder = Path("train_test")
    train_test_set_folder.mkdir(exist_ok=True)
    for idx, file in enumerate(list_file):
        pd.concat(lst[idx]).to_csv(train_test_set_folder / file)


def spliting_train_test(is_fimo: bool, is_cleaned: bool):
    csv_folder = get_csv_folder(is_fimo, is_cleaned)
    list_csv = list(csv_folder.glob("*.csv"))
    lst = [[], [], [], []]
    for csv in list_csv:
        data = pd.read_csv(csv, index_col=0)
        train_test_set = train_test_split(
            data.drop('Label', axis=1), data['Label'], test_size=0.4, random_state=123)
        for idx, set_ in enumerate(train_test_set):
            lst[idx].append(set_)
    train_test_names = get_name_train_test_set(is_fimo, is_cleaned)
    write_train_test_set_to_csv(lst, train_test_names)


def get_train_test_set(is_fimo: bool, is_cleaned: bool):
    files = get_name_train_test_set(is_fimo, is_cleaned)
    train_test_set = []
    train_test_set_folder = Path("train_test")
    for file in files:
        train_test_set.append(pd.read_csv(
            train_test_set_folder / file, index_col=0))
    return train_test_set
