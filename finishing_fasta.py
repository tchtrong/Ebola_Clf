# %%
from pathlib import Path
import glob
import re
import shutil

# %%
# Create all_*.fasta

# Get fasta and fasta_cleaned folders
fasta_folders = glob.glob("fasta*")
fasta_folders = [
    folder for folder in fasta_folders if "original" not in folder]

# Create motifs folder
p = Path('motifs')
p.mkdir(exist_ok=True)

for folder in fasta_folders:
    not_zai = glob.glob(folder + "/[!az]*.fasta")
    zais = glob.glob(folder + "/zai_*.fasta")
    not_zai.sort()
    for zai in zais:
        seed = re.search("\d+", zai).group(0)
        p.joinpath(seed).mkdir(exist_ok=True)
        with open(folder + "/all_{}.fasta".format(seed), 'wb') as wfd:
            files = not_zai.copy()
            files.append(zai)
            for f in files:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)

# %%
