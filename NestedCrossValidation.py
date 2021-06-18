# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre
# %%

from utils.common import DIR, get_folder
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation
from joblib import Memory
from shutil import rmtree
import pandas as pd
from pathlib import Path
from joblib import dump


# %%
"""
Data preparation
"""
csv_folder = get_folder(dir_type=DIR.CSV, no_X=True, fimo=True)
dataset = pd.read_csv(csv_folder/'all.csv', index_col=0)
X = dataset.drop('Label', axis=1)
y = dataset['Label']
random_state = 283258281
np.set_printoptions(precision=2, edgeitems=1)
pd.options.display.max_rows = 2
pd.options.display.max_columns = 2

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
    ('reduce_dim', LatentDirichletAllocation()),
    # ('scale_feats', 'passthrough'),
    ('classify', SVC()),
], memory=memory)

N_COMPONENTS = list(range(900, 1100, 10))
# N_COMPONENTS = [100]
C_OPTIONS = np.logspace(start=1, stop=1, num=1)
GAMMA_OPTIONS = np.logspace(start=-2, stop=-2, num=1)
SCALER_OPTIONS = [
    StandardScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    RobustScaler(),
    'passthrough',
]
KERNEL_OPTIONS = [
    # 'linear',
    'rbf',
]

param_grid = dict({
    'reduce_dim__n_components': N_COMPONENTS,
    # 'scale_feats': SCALER_OPTIONS,
    'classify__C': [1],
    'classify__gamma': [0.01],
    # 'classify__kernel': KERNEL_OPTIONS,
})

# Step 3: Set up cross-validation
inner_cv = StratifiedKFold(n_splits=3)
outer_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=10, random_state=random_state)

# Step 4: Set up GridSearchCV
clf = GridSearchCV(estimator=pipe, param_grid=param_grid,
                   cv=inner_cv)

# Step 5: Run nested cross-validation
cvv = cross_validate(clf, X, y, cv=outer_cv, return_estimator=True, n_jobs=-1)

dump(cvv, sol4/'cvv_LDA_900_1100_10_ik3_ir1_ok3_or20.bin')

memory.clear(warn=False)
rmtree(location)
