#%%
from global_variables import organisms
from Bio import SeqIO
import pandas as pd


#%%
#Read fasta
records = {}
sizes = []
for organism in organisms:
    record = list(SeqIO.parse("fasta_original/{}.fasta".format(organism), "fasta"))
    #Remove sequences contain X
    record = [sequence for sequence in record if str(sequence.seq).find("X") == -1]
    #Remove description of sequence (MEME will warn if the description is too long)
    for sequence in record:
        sequence.description = ""
    #Update sizes
    if organism == "zai":
        sizes.append(270)
    else:
        sizes.append(len(record))
    records.update({organism: record})

#Save size for each species
pd.DataFrame([sizes], columns=organisms).to_csv("fasta/sizes.csv")

print(sizes)

#%%
#Save fasta
for organism in organisms:
    SeqIO.write(records[organism], "fasta/{}.fasta".format(organism), "fasta")

#%%
#Remove duplicate sequences
def remove_dup_seqs(records, checksums):
    for record in records:
        checksum = record.seq
        if checksum in checksums:
            continue
        checksums.add(checksum)
        yield record

sizes = []
checksums = set()
for organism in organisms:
    cleaned = remove_dup_seqs(records[organism], checksums)
    if organism == "zai":
        sizes.append(75)
        SeqIO.write(cleaned, "fasta_cleaned/{}.fasta".format(organism), "fasta")
    else: 
        sizes.append(SeqIO.write(cleaned, "fasta_cleaned/{}.fasta".format(organism), "fasta"))

print(sizes)

#Save size for each species
pd.DataFrame([sizes], columns=organisms).to_csv("fasta_cleaned/sizes.csv")