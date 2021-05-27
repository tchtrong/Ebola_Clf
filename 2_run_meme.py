# %%

import subprocess
from pathlib import Path


def run_meme(no_X: bool):
    motifs_folder = Path()
    fasta = Path()
    if no_X:
        motifs_folder = Path("motifs_no_X")
        fasta = Path("fasta_cleaned_no_X")
    else:
        motifs_folder = Path("motifs")
        fasta = Path("fasta_cleaned")
    motifs_folder.mkdir(exist_ok=True)
    fasta = fasta.joinpath("all.fasta")
    for i in range(3, 21):
        sub_folder = motifs_folder.joinpath("{}".format(i))
        sub_folder.mkdir(exist_ok=True)
        process = subprocess.run(["/home/trongtch_vt/meme/bin/meme",
                                  "-searchsize", "0",
                                  "-protein",
                                  "-objfun", "classic",
                                  "-oc", sub_folder.as_posix(),
                                  "-mod", "zoops",
                                  "-minw", str(i), "-maxw", str(i),
                                  "-nmotifs", "150",
                                  "-minsites", "20",
                                  "-evt", "5.0e-006",
                                  "-p", '96 --use-hwthread-cpus',
                                  fasta.as_posix()], capture_output=True)
        if process.returncode != 0:
            print(process.stderr)
            break
        else:
            print("Finished finding motifs with length {}".format(i))


# %%
run_meme(no_X=True)
