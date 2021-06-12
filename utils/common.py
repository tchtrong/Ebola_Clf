from pathlib import Path
import shutil
from enum import Enum, unique
import pandas as pd


@unique
class DIR(Enum):
    DATA = "dataset"
    CSV = "csv"
    MOTIFS = "motifs"
    FIMO = "fimo"
    LABELS = "labels"
    SVM_TRAIN_TEST = "SVM_train_test"
    SVM_RESULTS = "SVM_results"
    LDA_MODEL = "LDA_models"
    LDA_TRAIN_TEST = "LDA_train_test"
    LDA_RESULTS = "LDA_results"


class MODEL(Enum):
    LDA = "LDA"


class FILE(Enum):
    SVM = "SVM"
    LDA_MODEL = "LDA"


class EXT(Enum):
    CSV = ".csv"
    NONE = ""


class SCLALER(Enum):
    RobustScaler = "robust"
    StandardScaler = "std"
    MinMaxScaler = "minmax"
    MaxAbsScaler = "maxabs"
    NONE = "org"


def get_folder(dir_type: DIR, no_X: bool,
               fimo: bool = False, recreate: bool = False) -> Path:

    folder = dir_type.value

    if folder != "motifs" and fimo:
        folder += "_fimo"
    if no_X:
        folder += "_no_X"

    folder = Path(folder)
    if recreate:
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True)

    return folder


def get_model(model_type: MODEL, n_comps: int, no_X: bool, fimo: bool):
    model_folder = None
    if model_type == MODEL.LDA:
        model_folder = get_folder(dir_type=DIR.LDA_MODEL, no_X=no_X, fimo=fimo)
        return model_folder/'LDA_train_only_{}'.format(n_comps)
