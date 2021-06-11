# %%
from utils.LDA import run_LDA

# %%
run_LDA(no_X=True, fimo=True, dimens=range(
    10, 11, 100), use_test=False, n_jobs=-1)
