# %%
import subprocess
from pathlib import Path


def run_fimo(path):
    p = Path(path)
    motif_folder = Path()
    if "X" in p.as_posix():
        motif_folder = Path("motifs_no_X")
    else:
        motif_folder = Path("motifs")
    list_fasta = list(p.glob("[!a]??.fasta"))
    fimo_dir = Path('fimo' + p.as_posix()[5:])
    fimo_dir.mkdir(exist_ok=True)
    for fasta in list_fasta:
        species = fasta.name[:3]
        species_dir = fimo_dir.joinpath(species)
        species_dir.mkdir(exist_ok=True)
        for i in range(3, 21):
            file_motif = motif_folder.joinpath(
                "{}_{}".format(i, i)).joinpath("meme.txt")
            out = subprocess.run(["fimo",
                                  "--skip-matched-sequence",
                                  "--max-stored-scores", "1000000",
                                  file_motif.as_posix(),
                                  fasta.as_posix()],
                                 capture_output=True)
            with open(species_dir / "{}.txt".format(i), "wb") as f:
                f.write(out.stdout)


# %%
run_fimo("fasta_no_X")
