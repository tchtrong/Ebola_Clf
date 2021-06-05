from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from itertools import compress
import shutil
import random
from utils.common import get_folder, DIR


def remove_duplicate_seq(records: list[SeqRecord]):
    store_uniques = set()
    new_records = []
    for record in records:
        if record.seq in store_uniques:
            continue
        else:
            store_uniques.add(record.seq)
            new_records.append(record)
    records[:] = new_records


def is_human_host(record: SeqRecord) -> bool:
    if 'host' in record.features[0].qualifiers:
        for host in record.features[0].qualifiers['host']:
            if 'Homo' in host:
                return True
    if 'isolate' in record.features[0].qualifiers:
        for host in record.features[0].qualifiers['isolate']:
            if 'H.sapiens' in host:
                return True
    return False


def remove_non_human_host_sequence(records: list[SeqRecord]):
    selectors = (is_human_host(record) for record in records)
    idx = 0
    for i, s in enumerate(compress(records, selectors)):
        records[i] = s
        idx = i
    if not idx:
        del records[idx:]
    else:
        del records[idx + 1:]


def remove_sequence_has_X(records: list[SeqRecord]):
    selectors = ('X' not in record.seq for record in records)
    idx = 0
    for i, s in enumerate(compress(records, selectors)):
        records[i] = s
        idx = i
    if not idx:
        del records[idx:]
    else:
        del records[idx + 1:]


def processing_records(records_path: Path, format: str, no_X: bool) -> list[SeqRecord]:
    records = list(SeqIO.parse(records_path, format))
    remove_non_human_host_sequence(records)
    remove_duplicate_seq(records)
    if no_X:
        remove_sequence_has_X(records)
    return records


def clean_records(records: list[SeqRecord], start_idx: list[int]):
    for idx, record in enumerate(records):
        record.id = str(start_idx[0] + idx)
        record.name = str(start_idx[0] + idx)
        record.description = ""
    start_idx[0] += len(records)


def random_zaire(out_folder: Path, seed: int, zai_idx: list[int]):
    file = out_folder / ("zai.fasta")
    file_copy = out_folder / ("zai_all.fasta")
    shutil.copy(file, file_copy)
    zais = list(SeqIO.parse(file, "fasta"))
    random.seed(seed)
    zais = random.sample(zais, 50)
    clean_records(zais, zai_idx)
    SeqIO.write(zais, file, "fasta")


def create_MEME_input(data_folder: Path):
    files = list(data_folder.glob("[!a]??[!_]*"))
    files.sort()
    with open(data_folder / "all.fasta", "wb") as fout:
        for file in files:
            with open(file, "rb") as fin:
                shutil.copyfileobj(fin, fout)


def processing_data(no_X: bool = True, seed: int = 423):
    data_folder = Path("dataset_org")
    files = list(data_folder.glob("*"))
    files.sort()
    format = files[0].name[4:]

    out_folder = get_folder(dir_type=DIR.DATA, no_X=no_X)

    idx = [0]
    zai_idx = [0]
    for file in files:
        records = processing_records(file, format, no_X)
        if "zai" in file.name:
            zai_idx[0] = idx[0]
        if len(records):
            clean_records(records, idx)
            print(file.stem + ".fasta")
            SeqIO.write(records, out_folder/(file.stem + ".fasta"), "fasta")

    random_zaire(out_folder, seed, zai_idx)
    create_MEME_input(out_folder)
