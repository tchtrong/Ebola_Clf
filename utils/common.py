from pathlib import Path
import shutil
from enum import Enum, unique


@unique
class DIR(Enum):
    DATA = "dataset"
    CSV = "csv"
    MOTIFS = "motifs"
    FIMO = "fimo"
    SVM = "SVM"
    LDA_MODEL = "LDA_models"


class FILE(Enum):
    SVM = "SVM"
    LDA_MODEL = "LDA"


class EXT(Enum):
    CSV = ".csv"
    NONE = ""


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


def get_file(file_type: FILE, ext: EXT = EXT.NONE, no_X: bool = True,
             fimo: bool = False, ith_comp: int = 0, use_test: bool = False) -> str:

    file_name = ''
    if not file_type is FILE.LDA_MODEL:
        file_name = file_type.value
        if fimo:
            file_name += "_fimo"
        if no_X:
            file_name += "_no_X"
        file_name += ext.value
    else:
        file_name = 'LDA'
        if use_test:
            file_name += "_train_test"
        else:
            file_name += "_train_only"
        file_name += str(ith_comp)

    return file_name
