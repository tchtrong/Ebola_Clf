# %%
from utils.LDA_SVM import run_SVM_LDA


# %%
run_SVM_LDA(no_X=True, fimo=True, dimens=range(10, 3360, 10), use_test=False)
