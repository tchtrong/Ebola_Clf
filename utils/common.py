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


class FILE(Enum):
    SVM = "SVM"


class EXT(Enum):
    CSV = ".csv"


def get_folder(dir_type: DIR, no_X: bool,
               fimo: bool = False, recreate: bool = False) -> Path:

    folder = dir_type.value

    if folder == "csv" and fimo:
        folder += "_fimo"
    if no_X:
        folder += "_no_X"

    folder = Path(folder)
    if recreate:
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True)

    return folder


def get_file(file_type: FILE, ext: EXT, no_X: bool,
             fimo: bool = False) -> str:

    file_name = file_type.value
    if fimo:
        file_name += "_fimo"
    if no_X:
        file_name += "_no_X"
    file_name += ext.value

    return file_name
