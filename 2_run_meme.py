# %%
from utils.handle_meme import run_meme

# %%
#run_meme("dataset_no_X", mode="oops", n_motifs=200, n_threads=4, searchfull=True)
run_meme("dataset_no_X", mode="anr", n_motifs=200, n_threads=12, searchfull=True)
