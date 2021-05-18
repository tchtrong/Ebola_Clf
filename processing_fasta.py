#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:15:57 2021

@author: binn
"""

#%%
from global_variables import organisms

from Bio import SeqIO

import random

#%%
records = {}

for organism in organisms:
    record = list(SeqIO.parse("fasta_original/{}.fasta".format(organism), "fasta"))
    record = [sequence for sequence in record if str(sequence.seq).find("X") == -1]
    if (organism == "zai"):
        random.seed(10)
        record = random.sample(record, 270)
    records.update({organism: record})

#%%
all_records = []
for organism in organisms:
    all_records = all_records + list(records[organism])
    SeqIO.write(records[organism], "fasta/{}.fasta".format(organism), "fasta")

SeqIO.write(all_records, "fasta/all.fasta", "fasta")
SeqIO.write(all_records, "motifs/all.fasta", "fasta")

# %%
