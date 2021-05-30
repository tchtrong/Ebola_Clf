# %%
from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from joblib import dump
import pandas as pd
import gc


# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    is_fimo=True, is_cleaned=True)

# %%
for i in range(1910, 2540, 100):
    lda_model = lda(n_components=i, random_state=43, n_jobs=-1)
    lda_model.fit(X_train)
    dump(lda_model, 'LDA_models/LDA_{}_train_only'.format(i))
    del lda_model
    gc.collect()
    lda_model = lda(n_components=i, random_state=43, n_jobs=-1)
    lda_model.fit(pd.concat([X_train, X_test]))
    dump(lda_model, 'LDA_models/LDA_{}_train_test'.format(i))
    del lda_model
    gc.collect()
