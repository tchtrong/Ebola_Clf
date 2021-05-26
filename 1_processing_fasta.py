# %%
from pathlib import Path
from utils import processing_fasta

# %%
fasta_orginal = Path("fasta_original")
list_fasta = list(fasta_orginal.glob("*"))
list_fasta.sort()

# %%
processing_fasta(list_fasta, keep_X=True, keep_duplicate=True)
processing_fasta(list_fasta, keep_X=True, keep_duplicate=False)
processing_fasta(list_fasta, keep_X=False, keep_duplicate=True)
processing_fasta(list_fasta, keep_X=False, keep_duplicate=False)

# %%
