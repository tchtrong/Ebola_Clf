from Bio import SeqIO
from pathlib import Path


def normalize_seq(seq_record, id):
    seq_record.id = str(id)
    seq_record.name = str(id)
    seq_record.description = ''


def read_fasta(path, keep_X, keep_duplicate, idx):
    check_duplicate = set()
    list_sequences = list()
    for seq_record in SeqIO.parse(path, "fasta"):
        if not keep_X and 'X' in seq_record.seq:
            continue
        if not keep_duplicate:
            if seq_record.seq in check_duplicate:
                continue
            else:
                check_duplicate.add(seq_record.seq)
        normalize_seq(seq_record, idx[0])
        list_sequences.append(seq_record)
        idx[0] += 1
    return list_sequences


def processing_fasta(list_fasta, keep_X, keep_duplicate):
    folder = "fasta"
    if not keep_duplicate:
        folder += "_cleaned"
    if not keep_X:
        folder += "_not_X"
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    idx = [0]
    for fasta in list_fasta:
        SeqIO.write(read_fasta(fasta, keep_X, keep_duplicate, idx),
                    folder / fasta.name, "fasta")
