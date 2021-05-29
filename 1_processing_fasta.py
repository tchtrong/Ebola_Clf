# %%
from pathlib import Path
from utils.utils import processing_fasta

# %%
fasta_orginal = Path("fasta_original")
list_fasta = list(fasta_orginal.glob("*"))
list_fasta.sort()

# %%
states = [True, False]
for keep_duplicate in states:
    for keep_X in states:
        processing_fasta(list_fasta, keep_X, keep_duplicate)

# %%
