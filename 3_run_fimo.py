# %%
import subprocess
from pathlib import Path
import shutil


def run_fimo(data_folder, motif_folder, thresh: float = 1.0e-4):

    data_folder = Path(data_folder)
    motif_folder = Path(motif_folder)

    files = list(data_folder.glob("[!a]??[!_]*"))

    fimo_dir = Path('fimo' + data_folder.as_posix()[7:])
    shutil.rmtree(fimo_dir, ignore_errors=True)
    fimo_dir.mkdir(exist_ok=True)

    for file in files:
        species = file.name[:3]
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


# %%
run_fimo("dataset_no_X", "motifs_no_X")
