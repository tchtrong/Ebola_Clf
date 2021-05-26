# %%
from pathlib import Path
import glob
import pandas as pd
import re
import shutil


def read_fimo(path):
    data = pd.read_csv(path, sep='\t')
    motifs = data['motif_id']
    sequences = data['sequence_name']
    list = [{}]
    last_motif = motifs[0]
    for idx, sequence in enumerate(sequences):
        if last_motif != motifs[idx]:
            list.append({})
        if sequence not in list[-1]:
            list[-1][sequence] = 1
        else:
            list[-1][sequence] += 1
        last_motif = motifs[idx]
    list = pd.DataFrame(list, index=motifs.unique())
    return list


# %%
p = Path("fimo")
list_fasta = list(Path("fasta").glob("[!az]*.fasta"))
list_fasta = list_fasta + list(Path("fasta").glob("zai_*.fasta"))

folder_list = []
for fasta in list_fasta:
    folder_list.append(fasta.name[:-6])

folder_list.sort()

# %%
freq_all = pd.DataFrame()
for folder in folder_list:
    cur = p.joinpath(folder)
    freq_matrix = pd.DataFrame()
    for i in range(3, 21):
        i_matrix = read_fimo(cur.joinpath(str(i)).as_posix() + '.txt')
        freq_matrix = pd.concat([freq_matrix, i_matrix])
    freq_matrix.T.fillna(0).to_csv('csv_fimo/{}.csv'.format(folder))
    if "zai" in cur.as_posix():
        pd.concat([freq_all, freq_matrix.T.fillna(0)]).to_csv(
            'csv_fimo/all_{}.csv'.format(folder[-2:]))
    else:
        freq_all = pd.concat([freq_all, freq_matrix.T.fillna(0)])

# %%
