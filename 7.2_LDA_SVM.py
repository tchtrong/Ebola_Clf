# %%
from utils.SVM import run_SVM_LDA


# %%
run_SVM_LDA(no_X=True, fimo=True, topic_range=range(100, 3400, 100), kernel='linear')
