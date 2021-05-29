# %%
from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from joblib import dump
import pandas as pd


# %%
X_train, X_test, y_train, y_test = get_train_test_set(
    is_fimo=True, is_cleaned=True)

# %%
for i in range(10, 2540, 10):
    lda_model = lda(n_components=i, random_state=43, n_jobs=-1)
    lda_model.fit(X_train)
    dump(lda_model, 'LDA_models/LDA_{}_train_only'.format(i))
    lda_model.fit(pd.concat([X_train, X_test]))
    dump(lda_model, 'LDA_models/LDA_{}_train_test'.format(i))