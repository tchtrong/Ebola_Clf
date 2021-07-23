# %%

from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD as LSA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

from utils.common import DIR, get_folder
from utils.fstm import FSTM
from utils.plsa import PLSA


def get_n_top_words(model, n_top_words):
    words = set()
    feature_names = list(range(0, 3351, 1))
    for topic in model.components_:
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        for i in top_features_ind:
            words.add(str(feature_names[i]))
    return words


def update_result(fit_time: np.ndarray, score_time: np.ndarray, n_components: int, total_runs: int, C: int, gamma: int, results: pd.DataFrame, row_ind: int,
                  N_TOP_WORDS: int = None, idx_tw: int = None):
    scores = results.iloc[row_ind, 8:8+total_runs]
    results['mean_test_score'][row_ind] = scores.mean()
    results['std_test_score'][row_ind] = scores.std()
    results['mean_fit_time'].iloc[row_ind] = fit_time.mean()
    results['std_fit_time'].iloc[row_ind] = fit_time.std()
    results['mean_score_time'].iloc[row_ind] = score_time.mean()
    results['std_score_time'].iloc[row_ind] = score_time.std()
    results['n_components'].iloc[row_ind] = n_components
    if idx_tw is None:
        results['n_top_motifs'].iloc[row_ind] = 0
    else:
        results['n_top_motifs'].iloc[row_ind] = N_TOP_WORDS[idx_tw]
    results['C'].iloc[row_ind] = C
    results['gamma'].iloc[row_ind] = gamma


def run_SVM(total_runs, train_test_sets, C, gamma, n_components, results, row_ind, N_TOP_WORDS: int = None, idx_tw: int = None, is_linear=False):
    fit_time = np.zeros(total_runs)
    score_time = np.zeros(total_runs)

    for idx_tts, (X_train_new, X_test_new, y_train, y_test) in enumerate(train_test_sets):

        if is_linear:
            clf = SVC(kernel='linear', C=C, gamma=gamma)
        else:
            clf = SVC(C=C, gamma=gamma)

        start = time()
        clf.fit(X_train_new, y_train)
        fit_time[idx_tts] = time() - start

        start = time()
        score = clf.score(X_test_new, y_test)
        score_time[idx_tts] = time() - start

        results.iloc[row_ind, 8+idx_tts] = score

    if idx_tw is None:
        update_result(fit_time, score_time, n_components,
                      total_runs, C, gamma, results, row_ind)
    else:
        update_result(fit_time, score_time, n_components,
                      total_runs, C, gamma, results, row_ind, N_TOP_WORDS, idx_tw)


# %%
"""
Data preparation
"""
csv_folder = get_folder(dir_type=DIR.CSV, no_X=True, fimo=True)
dataset = pd.read_csv(csv_folder/'all.csv', index_col=0)
X = dataset.drop('Label', axis=1)
y = dataset['Label']
random_state = 283258281

'''
Train test set preparation
'''
n_splits = 3
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# %%
"""
Params preparation
"""
C_OPTIONS = np.logspace(-7, 15, 23, base=2)
GAMMA_OPTIONS = np.logspace(-15, 7, 23, base=2)
N_COMPONENTS = [5, 10, 20, 50, 100, 200, 400, 800, 1500]
N_TOP_WORDS = [3, 5, 7, 10]

# %%
'''
Results preparation
'''
total_runs = n_splits * n_repeats

split_test_scores = [
    'split{}_test_score'.format(i) for i in range(total_runs)
]

split_motifs = [
    'split{}_motifs'.format(i) for i in range(total_runs)
]


labels_results = ['mean_fit_time'] + ['std_fit_time'] + ['mean_score_time'] + ['std_score_time'] +\
    ['n_components', 'n_top_motifs', 'C', 'gamma'] + \
    split_test_scores + ['mean_test_score'] + ['std_test_score']

labels_top_motifs = ['n_components', 'n_top_motifs'] + split_motifs

labels_model_timming = ['n_components', 'mean_fit_time', 'std_fit_time',
                        'mean_trans_train_time', 'std_trans_train_time', 'mean_trans_test_time', 'std_trans_test_time']

n_rows_results = len(N_COMPONENTS) * (len(N_TOP_WORDS) + 1) * \
    len(C_OPTIONS) * len(GAMMA_OPTIONS)

n_rows_top_motifs = len(N_COMPONENTS) * len(N_TOP_WORDS)

n_rows_timming = len(N_COMPONENTS)

results = np.zeros(shape=(n_rows_results, len(labels_results)))
top_motifs = np.zeros(shape=(n_rows_top_motifs, len(labels_top_motifs)))
timming = np.zeros(shape=(n_rows_timming, len(labels_model_timming)))

results = pd.DataFrame(results, columns=labels_results)
top_motifs = pd.DataFrame(top_motifs, columns=labels_top_motifs, dtype=object)
timming = pd.DataFrame(timming, columns=labels_model_timming)

# %%
'''
Run models
'''
row_ind = 0
for idx_n, n_components in enumerate(N_COMPONENTS):

    train_test_sets = [0] * total_runs
    train_test_top_words_sets = [0] * total_runs
    train_test_top_words_sets = [train_test_top_words_sets] * len(N_TOP_WORDS)

    fit_time = np.zeros(total_runs)
    trans_train_time = np.zeros(total_runs)
    trans_test_time = np.zeros(total_runs)

    for idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train = np.require(X_train, requirements='C')
        X_test = np.require(X_test, requirements='C')

        model = FSTM(n_components=n_components,
                     random_state=random_state)

        start = time()
        model.fit(X_train)
        fit_time[idx] = time() - start

        start = time()
        X_train_new = model.transform(X_train)
        trans_train_time[idx] = time() - start

        start = time()
        X_test_new = model.transform(X_test)
        trans_test_time[idx] = time() - start

        train_test_sets[idx] = (X_train_new, X_test_new, y_train, y_test)

        for idx_tw, n in enumerate(N_TOP_WORDS):
            top_words = get_n_top_words(model, n)

            top_motifs.iloc[idx_n*len(N_TOP_WORDS) +
                            idx_tw, 2 + idx] = top_words
            top_motifs.iloc[idx_n*len(N_TOP_WORDS) +
                            idx_tw, 0] = n_components
            top_motifs.iloc[idx_n*len(N_TOP_WORDS) +
                            idx_tw, 1] = n

            X_train_new = X_train[top_words]
            X_test_new = X_test[top_words]

            train_test_top_words_sets[idx_tw][idx] = (
                X_train_new, X_test_new, y_train, y_test)

    timming['mean_fit_time'].loc[idx_n] = fit_time.mean()
    timming['std_fit_time'].loc[idx_n] = fit_time.std()
    timming['mean_trans_train_time'].loc[idx_n] = trans_train_time.mean()
    timming['std_trans_train_time'].loc[idx_n] = trans_train_time.std()
    timming['mean_trans_test_time'].loc[idx_n] = trans_test_time.mean()
    timming['std_trans_test_time'].loc[idx_n] = trans_test_time.std()
    timming['n_components'].loc[idx_n] = n_components

    for idx_c, C in enumerate(C_OPTIONS):
        for idx_g, gamma in enumerate(GAMMA_OPTIONS):
            run_SVM(total_runs, train_test_sets, C,
                    gamma, n_components, results, row_ind)
            row_ind += 1

    for idx_tw, tw in enumerate(train_test_top_words_sets):
        for idx_c, C in enumerate(C_OPTIONS):
            for idx_g, gamma in enumerate(GAMMA_OPTIONS):
                run_SVM(total_runs, tw, C, gamma, n_components,
                        results, row_ind, N_TOP_WORDS, idx_tw)
                row_ind += 1

# %%
rs_path = Path('results')
rs_path.mkdir(exist_ok=True)

dump(results, rs_path/'FSTM_results_rbf.bin')
dump(timming, rs_path/'FSTM_timming.bin')
dump(top_motifs, rs_path/'FSTM_top_motifs.bin')

# %%