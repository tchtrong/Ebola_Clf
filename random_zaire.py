#%%
from Bio import SeqIO
import random

#%%
#Create random seeds
random.seed(951)
seeds = random.sample(range(100), 6)

#%%
#Random zaire
zaire = list(SeqIO.parse("fasta/zai.fasta", "fasta"))
zaire_cleaned = list(SeqIO.parse("fasta_cleaned/zai.fasta", "fasta"))

for seed in seeds:
    random.seed(seed)
    SeqIO.write(random.sample(zaire, 270), "fasta/zai_{}.fasta".format(seed),"fasta")
    SeqIO.write(random.sample(zaire_cleaned, 75), "fasta_cleaned/zai_{}.fasta".format(seed),"fasta")

# %%
