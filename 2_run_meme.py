# %%
from utils.meme import run_meme

# %%
run_meme(no_X=True, mode="zoops", n_motifs=1000, length_range=range(3, 21), n_threads=12, searchfull=True)
