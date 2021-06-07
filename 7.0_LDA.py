# %%
from utils.LDA import run_LDA

# %%
run_LDA(no_X=True, fimo=True, dimens=range(10, 3360, 10), n_jobs=-1)
