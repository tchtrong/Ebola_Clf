# %%
import subprocess
from pathlib import Path

# %%
p = Path(".")
fimo_dir = p.joinpath("fimo")
fimo_dir.mkdir(exist_ok=True)

# %%
list_fasta = list(p.joinpath("fasta").glob("[!az]*.fasta"))
list_fasta = list_fasta + list(p.joinpath("fasta").glob("zai_*.fasta"))

# %%
for fasta in list_fasta:
    fimo_dir.joinpath(fasta.name[:-6]).mkdir(exist_ok=True)
    for i in range(3, 21):
        out = subprocess.run(["fimo", "--skip-matched-sequence",
                        "motifs/{}_{}/meme.txt".format(i, i),
                        fasta.as_posix()], capture_output=True)
        with open("fimo/{}/{}.txt".format(fasta.name[:-6], i), "wb") as f:
            f.write(out.stdout)
