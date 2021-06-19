# %%

from utils.common import DIR, get_folder
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import Memory
from shutil import rmtree
import pandas as pd
from pathlib import Path
from joblib import dump, parallel_backend, load
from sklearn.base import TransformerMixin, BaseEstimator
import time


class ReduceDimLDA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 100, n_top_words: int = 5):
        self.n_components = n_components
        self.model = LDA(n_components=self.n_components)
        self.n_top_words = n_top_words
        self.top_words = 0
        self.list_words = set()

    def get_n_top_words(self):
        words = set()
        feature_names = list(range(0, 3351, 1))
        for topic in self.model.components_:
            top_features_ind = topic.argsort()[:-self.n_top_words - 1:-1]
            for i in top_features_ind:
                words.add(str(feature_names[i]))
        return words

    def fit(self, X, y=None):
        self.model.fit(X)
        self.list_words = self.get_n_top_words()
        self.top_words = len(self.list_words)
        return self

    def transform(self, X, y=None):
        top_words = self.list_words
        return X[top_words]


# %%
"""
Data preparation
"""
csv_folder = get_folder(dir_type=DIR.CSV, no_X=True, fimo=True)
dataset = pd.read_csv(csv_folder/'all.csv', index_col=0)
X = dataset.drop('Label', axis=1)
y = dataset['Label']
random_state = 283258281

# %%
"""
Solution 4: Nested cross-validation
"""
# Step 1: Create folder to store result
sol4 = Path('sol4')
sol4.mkdir(exist_ok=True)

# Step 2: Set up Pipeline for optimizing hyperparameters
location = 'cachedir'
memory = Memory(location=location, verbose=0)

pipe = Pipeline([
    ('reduce_dim', 'passthrough'),
    ('classify', SVC()),
], memory=memory)

N_COMPONENTS = list(range(10, 210, 10))
N_TOP_WORDS = list(range(1, 11, 1))
C_OPTIONS = np.logspace(start=0, stop=0, num=1)
GAMMA_OPTIONS = np.logspace(start=-3, stop=-3, num=1)
KERNEL_OPTIONS = [
    'linear',
    # 'rbf',
]

param_grid = [
    # {
    #     'reduce_dim': ['passthrough'],
    #     'classify__C': C_OPTIONS,
    #     'classify__gamma': GAMMA_OPTIONS,
    #     'classify__kernel': KERNEL_OPTIONS,
    # },
    {
        'reduce_dim': [ReduceDimLDA()],
        'reduce_dim__n_components': N_COMPONENTS,
        'reduce_dim__n_top_words': N_TOP_WORDS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
        'classify__kernel': KERNEL_OPTIONS,
    },
]

# Step 3: Set up cross-validation
inner_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=10, random_state=random_state)
outer_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=2, random_state=random_state)

# Step 4: Set up GridSearchCV
clf = GridSearchCV(estimator=pipe, param_grid=param_grid,
                   cv=inner_cv)

with parallel_backend('threading', n_jobs=4):
    clf.fit(X, y)
dump(clf, sol4/'clf_LDA_10.210.10_1.11.1_k3r10.bin')


# Step 5: Run nested cross-validation
# with parallel_backend('threading', n_jobs=4):
#     cvv = cross_validate(clf, X, y, cv=outer_cv, return_estimator=True)
# dump(cvv, sol4/'cvv_LDA_50.60.10_7.8.1_ik3_ir1_ok3_or2.bin')

memory.clear(warn=False)
rmtree(location)

# %%
# for es in cvv['estimator']:
#     print(es.best_params_)
# %%
