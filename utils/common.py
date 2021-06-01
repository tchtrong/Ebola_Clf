from pathlib import Path
import shutil


def get_folder(folder, no_X: bool, fimo: bool = False, recreate: bool = False):
    if folder == "csv" and fimo:
        folder += "_fimo"
    if no_X:
        folder += "_no_X"

    folder = Path(folder)
    if recreate:
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True)

    return folder
