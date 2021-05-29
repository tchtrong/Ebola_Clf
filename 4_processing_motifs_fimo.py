# %%
from pathlib import Path
import pandas as pd
from Bio import SeqIO


def read_fimo(path, n_seq, n_motifs, n_last_seqs):

    data = pd.read_csv(path, sep='\t')
    motifs = data['motif_alt_id']
    sequences = data['sequence_name']

    motif_range = list(range(1, n_motifs + 1))
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

    lst = pd.DataFrame(lst, dtype=pd.Int64Dtype)
    for i in seq_range:
        lst[i] = pd.Series(dtype=pd.Float64Dtype)

    return lst


def get_num_seq(species: Path, fimo_folder: Path):
    fasta_folder = Path("fasta" + str(fimo_folder)[4:])
    species = species.name + ".fasta"
    return len(list(SeqIO.parse(fasta_folder / species, "fasta")))


def get_num_motif(length: int):
    motif_folder = Path("motifs_no_X").joinpath(str(length))
    n_motifs = len(list(motif_folder.glob("*.eps")))
    return n_motifs


def processing_motifs(fimo_folder: Path):
    fimo_folder = Path(fimo_folder)

    csv_folder = Path("csv_" + str(fimo_folder))
    csv_folder.mkdir(exist_ok=True)

    species_list = list(fimo_folder.glob("*"))
    species_list.sort()

    n_last_seqs = 0
    for idx, species in enumerate(species_list):

        n_seq = get_num_seq(species, fimo_folder)
        freq_matrix = pd.DataFrame()

        for i in range(3, 21):
            n_motifs = get_num_motif(i)
            i_matrix = read_fimo(str(species.joinpath(
                str(i))), n_seq, n_motifs, n_last_seqs)
            freq_matrix = pd.concat([freq_matrix, i_matrix])

        to_csv = freq_matrix.T.fillna(0)
        to_csv['Label'] = pd.Series([idx] * freq_matrix.shape[0])
        to_csv.to_csv(csv_folder / '{}.csv'.format(species.name))

        n_last_seqs += n_seq


# %%
processing_motifs("fimo_cleaned_no_X")
processing_motifs("fimo_no_X")
# %%
