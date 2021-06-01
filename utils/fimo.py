from pathlib import Path
import pandas as pd
from Bio import SeqIO
from utils.common import get_folder
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt


def read_fimo(path, n_seq, n_motifs, n_last_seqs, n_last_motifs):

    data = pd.read_csv(path, sep='\t')
    motifs = data['motif_alt_id']
    sequences = data['sequence_name']

    motif_range = list(range(n_last_motifs, n_last_motifs + n_motifs))
    seq_range = set(range(n_last_seqs, n_last_seqs + n_seq))

    lst = []
    last_motif = ""
    for idx, sequence in enumerate(sequences):

        if sequence in seq_range:
            seq_range.remove(sequence)

        if last_motif != motifs[idx]:
            lst.append({})
            while len(lst) != int(motifs[idx][5:]):
                lst.append({})
            last_motif = motifs[idx]

        if sequence not in lst[-1]:
            lst[-1][sequence] = 1
        else:
            lst[-1][sequence] += 1

    while len(lst) != n_motifs:
        lst.append({})

    lst = pd.DataFrame(lst, dtype=pd.Int64Dtype, index=motif_range)
    for i in seq_range:
        lst[i] = pd.Series(dtype=float)

    return lst


def get_num_seq(species: Path, fimo_folder: Path):
    data_folder = Path("dataset" + str(fimo_folder)[4:])
    species = species.name + ".fasta"
    return len(list(SeqIO.parse(data_folder / species, "fasta")))


def get_num_motif(length: int, motifs_folder: Path):
    length_folder = motifs_folder.joinpath(str(length))
    n_motifs = len(list(length_folder.glob("*.eps")))
    return n_motifs


def remove_duplicate_vector(all_matrices: pd.DataFrame):
    all_matrices.drop_duplicates(
        subset=all_matrices.columns[:-1], keep=False, inplace=True)


def processing_motifs(no_X: bool):

    fimo_folder = get_folder("fimo", no_X)
    csv_folder = get_folder("csv", no_X, fimo=True, recreate=True)
    motifs_folder = get_folder("motifs", no_X)

    species_list = list(fimo_folder.glob("*"))
    species_list.sort()

    lst_matrices = []
    n_last_seqs = 0
    print("Before remove duplicate vectors:")
    for idx, species in enumerate(species_list):

        n_seqs = get_num_seq(species, fimo_folder)
        n_last_motifs = 0
        freq_matrix = []

        for i in range(3, 21):
            n_motifs = get_num_motif(i, motifs_folder)

            i_matrix = read_fimo(str(species.joinpath(
                str(i))), n_seqs, n_motifs, n_last_seqs, n_last_motifs)
            freq_matrix.append(i_matrix)
            n_last_motifs += n_motifs

        lst_matrices.append(pd.concat(freq_matrix).T.fillna(0))
        lst_matrices[-1]['Label'] = [idx] * n_seqs

        n_last_seqs += n_seqs

    lst_matrices = pd.concat(lst_matrices)

    lst_matrices.drop_duplicates(
        subset=lst_matrices.columns[:-1], keep=False, inplace=True)

    grouped = lst_matrices.groupby("Label")
    for idx, species in enumerate(species_list):
        matrix = grouped.get_group(idx)
        print(species, matrix.shape)
        matrix.to_csv(csv_folder / '{}.csv'.format(species.name))
