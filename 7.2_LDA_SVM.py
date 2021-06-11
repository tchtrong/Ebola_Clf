# %%
from utils.LDA_SVM import run_SVM_LDA


# %%
run_SVM_LDA(no_X=True, fimo=True, dimens=range(100, 2000, 100), use_test=False)
