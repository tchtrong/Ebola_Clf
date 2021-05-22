#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:45:04 2021

@author: binn

"""

#%%
#Read fasta files
from Bio import SeqIO
import xml.etree.ElementTree as ET
import re
from Bio.SeqIO.FastaIO import FastaTwoLineParser
import pandas as pd
import numpy as np

#%%
#Records
seed = 12

sequences = [str(sequence.seq) for sequence in list(SeqIO.parse("fasta/all_{}.fasta".format(seed), "fasta"))]

#%%
#Motifs
motifs = []
for idx in range(3, 21):
    for child in ET.parse("motifs/{}/{}_{}/meme.xml".format(seed, idx, idx)).getroot()[2]:
        motif = child[2].text.replace("\n", "")
        motifs.append(motif)
        
#%%
freq_matrix = []
for sequence in sequences:
    f_motifs_in_sequence = []
    for motif in motifs:
        f_motifs_in_sequence.append(len(re.findall(motif.replace("X", "[A-Z]"), sequence)))
    freq_matrix.append(f_motifs_in_sequence)
    
#%%
freq_matrix_csv = pd.DataFrame(freq_matrix, columns=(motifs))

#%%
#sum_of_freq_each_motif = list(freq_matrix_csv.sum().array)

#%%
#freq_matrix_csv.drop(columns=([motif for idx, motif in enumerate(motifs) if sum_of_freq_each_motif[idx] == 0]), inplace=(True))

#%%
freq_matrix_csv.to_csv("csv/freq_matrix.csv", index=False)

#%%
freq_matrix_without_X = freq_matrix_csv.drop(columns=[motif for motif in list(freq_matrix_csv.columns.values) if motif.count("X") != 0])
freq_matrix_without_X.to_csv("csv/freq_matrix_without_X.csv", index=False)

#%%
freq_matrix_with_X_less_than_half = freq_matrix_csv.drop(columns=[motif for motif in list(freq_matrix_csv.columns.values) if motif.count("X") >= round(len(motif) / 2)])
freq_matrix_with_X_less_than_half.to_csv("csv/freq_matrix_with_X_less_than_half.csv", index=False)



























