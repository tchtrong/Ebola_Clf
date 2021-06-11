import subprocess
from pathlib import Path
import shutil
from utils.common import get_folder, DIR


def run_fimo(no_X: bool, thresh: float = 1.0e-4):

    data_folder  = get_folder(dir_type=DIR.DATA, no_X=no_X)
    motif_folder = get_folder(dir_type=DIR.MOTIFS, no_X=no_X)
    fimo_dir     = get_folder(dir_type=DIR.FIMO, no_X=no_X, recreate=True)

    files = list(data_folder.glob("[!a]???[!a]*"))
    files.sort()

    for file in files:
        species = file.name[:-6]
        species_dir = fimo_dir.joinpath(species)
        species_dir.mkdir(exist_ok=True)

        for i in range(3, 21):
            file_motif = motif_folder.joinpath(
                "{}".format(i)).joinpath("meme.txt")

            out = subprocess.run(["fimo",
                                  "--skip-matched-sequence",
                                  "--max-stored-scores", "1000000",
                                  "--thresh", str(thresh),  # Default 1.0e-4
                                  file_motif.as_posix(),
                                  file.as_posix()],
                                 capture_output=True)

            if out.returncode == 1:
                print(out.stderr)
                break
            else:
                with open(species_dir / "{}".format(i), "wb") as f:
                    f.write(out.stdout)
