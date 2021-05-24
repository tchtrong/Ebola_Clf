# %%
from global_variables import organisms
from Bio import SeqIO
import random
from pathlib import Path
import glob
import re
import shutil


def update_seq_id(records, start_idx):
    for idx, record in enumerate(records):
        record.id = str(start_idx + idx)
        record.name = str(start_idx + idx)


# Global variables
random.seed(951)
seeds = random.sample(range(100), 6)

# %%
# Read fasta
idx = 0
idx_cleaned = 0
zai_idx = 0
zai_idx_cleaned = 0
all = open("fasta/all.fasta", "w")
all_cleaned = open("fasta_cleaned/all.fasta", "w")
for organism in organisms:
    check_dups = set()
    records = []
    records_cleaned = []
    if organism == "zai":
        zai_idx = idx
        zai_idx_cleaned = idx_cleaned
    for record in SeqIO.parse("fasta_original/{}.fasta".format(organism), "fasta"):
        # Not include seq which has X
        if "X" not in record.seq:
            record.description = ""
            record.name = str(idx)
            record.id = str(idx)
            records.append(record)
            idx += 1
            if record.seq not in check_dups:
                check_dups.add(record.seq)
                record_cp = record[:]
                record_cp.name = str(idx_cleaned)
                record_cp.id = str(idx_cleaned)
                records_cleaned.append(record_cp)
                idx_cleaned += 1
    SeqIO.write(records, "fasta/{}.fasta".format(organism), "fasta")
    SeqIO.write(records_cleaned,
                "fasta_cleaned/{}.fasta".format(organism), "fasta")
    SeqIO.write(records, all, "fasta")
    SeqIO.write(records_cleaned, all_cleaned, "fasta")

    if organism == "zai":
        for seed in seeds:
            random.seed(seed)

            zaire = random.sample(records, 270)
            zaire_cleaned = random.sample(records_cleaned, 75)

            update_seq_id(zaire, zai_idx)
            update_seq_id(zaire_cleaned, zai_idx_cleaned)

            SeqIO.write(zaire, "fasta/zai_{}.fasta".format(seed), "fasta")
            SeqIO.write(zaire_cleaned,
                        "fasta_cleaned/zai_{}.fasta".format(seed), "fasta")

all.close()
all_cleaned.close()

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
