# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre
# %%

from utils.common import DIR, SCLALER, get_folder
from utils.processing_train_test import get_labels, get_matrices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from joblib import Memory
from shutil import rmtree
import pandas as pd
from pathlib import Path
from joblib import dump, load


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
memory = Memory(location=location)

pipe = Pipeline([
    ('reduce_dim', LatentDirichletAllocation(n_jobs=8)),
    ('scale_feats', 'passthrough'),
    ('classify', SVC()),
], memory=memory)

N_COMPONENTS = list(range(500, 600, 100))
C_OPTIONS = np.logspace(start=-3, stop=3, num=7)
GAMMA_OPTIONS = np.logspace(start=-3, stop=3, num=7)
SCALER_OPTIONS = [#'passthrough',
                  MinMaxScaler(),
                  #StandardScaler(),
                  #MaxAbsScaler(),
                  #RobustScaler(),
                  ]

param_grid = dict({'reduce_dim__n_components': N_COMPONENTS,
                   'scale_feats': SCALER_OPTIONS,
                   'classify__C': C_OPTIONS,
                   'classify__gamma': GAMMA_OPTIONS})

# Step 3: Set up cross-validation
# inner_cv = StratifiedKFold(n_splits=3)
inner_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=2, random_state=random_state)
outer_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=10, random_state=random_state)

# Step 4: Set up GridSearchCV
clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, n_jobs=2)
clf.fit(X, y)

# Step 5: Run nested cross-validation
cvv = cross_validate(clf, X, y, cv=outer_cv, return_estimator=True, n_jobs=-1)

dump(cvv, sol4/'cvv.bin')

# %%
# Step 2: Set up Pipeline for LDA
pipe_svm = Pipeline([('classify', SVC())], memory=memory)

param_grid_svm = dict({'classify__C': C_OPTIONS,
                       'classify__gamma': GAMMA_OPTIONS})

# Step 3: Set up cross-validation
inner_cv_svm = StratifiedKFold(n_splits=3)
outer_cv_svm = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=10, random_state=random_state)

# Step 4: Set up GridSearchCV
clf_svm = GridSearchCV(
    estimator=pipe_svm, param_grid=param_grid_svm, cv=inner_cv_svm, n_jobs=-1)
cvv_svm = cross_validate(clf_svm, X, y, cv=outer_cv_svm,
                         return_estimator=True, n_jobs=-1)

dump(cvv_svm, sol4/'cvv_svm.bin')

memory.clear(warn=False)
rmtree(location)

# %%
