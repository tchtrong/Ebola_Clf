from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import shutil
from numpy.random import default_rng
import numpy as np


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


def concanate_fasta(path: Path):
    all_fasta = path.joinpath("all.fasta")
    all_fasta.unlink(missing_ok=True)
    for i in path.glob("zai_*"):
        i.unlink()
    list_fasta = list(path.glob("*.fasta"))
    list_fasta.sort()
    with open(all_fasta, 'wb') as wfd:
        for f in list_fasta:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def random_zaire(path: Path):
    num_zai_rand = 0
    if "cleaned" in path.as_posix():
        num_zai_rand = 75
    else:
        num_zai_rand = 270
    zai = path / "zai.fasta"
    shutil.copyfile(zai, path / "zai_all.fasta")
    list_zai = np.array(list(SeqIO.parse(zai, "fasta")), dtype=SeqRecord)
    idx = int(list_zai[0].id)
    rgn = default_rng(951)
    list_zai = rgn.choice(list_zai, num_zai_rand)
    for zai_ in list_zai:
        normalize_seq(zai_, idx)
        idx += 1
    SeqIO.write(list_zai, zai, "fasta")


def processing_fasta(list_fasta, keep_X, keep_duplicate):
    folder = "fasta"
    if not keep_duplicate:
        folder += "_cleaned"
    if not keep_X:
        folder += "_no_X"
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    idx = [0]
    for fasta in list_fasta:
        SeqIO.write(read_fasta(fasta, keep_X, keep_duplicate, idx),
                    folder / fasta.name, "fasta")
    concanate_fasta(folder)
    random_zaire(folder)
