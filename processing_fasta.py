# %%
import glob
import shutil
import re
from global_variables import organisms
from Bio import SeqIO
import pandas as pd
from pathlib import Path


def remove_dup_seqs(records):
    checksums = set()
    for record in records:
        checksum = record.seq
        if checksum in checksums:
            continue
        checksums.add(checksum)
        yield record


# %%
# Read fasta
sizes = {"": [], "_cleaned": []}
for organism in organisms:
    record = list(SeqIO.parse(
        "fasta_original/{}.fasta".format(organism), "fasta"))

    # Remove sequences contain X
    record = [sequence for sequence in record if str(
        sequence.seq).find("X") == -1]

    # Remove description of sequence (MEME will warn if the description is too long)
    for sequence in record:
        sequence.description = ""
        # sequence.name = idx
        # sequence.id = idx

    # Remove duplicate sequences
    cleaned = remove_dup_seqs(record)

    # Update sizes
    if organism == "zai":
        sizes[""].append(270)
        sizes["_cleaned"].append(75)
    else:
        sizes[""].append(SeqIO.write(
            record, "fasta/{}.fasta".format(organism), "fasta"))
        sizes["_cleaned"].append(SeqIO.write(
            cleaned, "fasta_cleaned/{}.fasta".format(organism), "fasta"))

# %%
# Save size for each species and save the labels
for key, values in sizes.items():
    labels = []
    for idx, value in enumerate(values):
        labels = labels + [idx] * value
    pd.DataFrame(labels).to_csv("csv/labels{}.csv".format(key), index=False)
    pd.DataFrame(sizes[key], index=organisms).to_csv(
        "fasta{}/sizes.csv".format(key), index=False)

