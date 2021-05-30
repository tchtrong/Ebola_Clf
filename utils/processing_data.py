from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from itertools import compress
import shutil


def remove_duplicate(records: list[SeqRecord]):
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


def not_human_host(record: SeqRecord) -> bool:
    return not is_human_host(record)


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


def processing_records(records_path: Path, format: str, keep_X: bool) -> list[SeqRecord]:
    records = list(SeqIO.parse(records_path, format))
    remove_non_human_host_sequence(records)
    remove_duplicate(records)
    if not keep_X:
        remove_sequence_has_X(records)
    return records


def processing_data(keep_X: bool = True):
    data_folder = Path("dataset_org")
    files = list(data_folder.glob("*"))
    format = files[0].name[4:]

    out_folder = "dataset"
    if not keep_X:
        out_folder += "_no_X"
    out_folder = Path(out_folder)
    shutil.rmtree(out_folder, ignore_errors=True)
    out_folder.mkdir()

    for file in files:
        records = processing_records(file, format, keep_X)
        if len(records):
            SeqIO.write(records, out_folder/file.name, format)
