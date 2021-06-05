# %%
from utils.LDA import run_LDA

# %%
run_LDA(no_X=True, fimo=True, dimens=range(100, 500, 50), n_jobs=-1)
