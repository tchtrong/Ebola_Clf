from sklearn.decomposition import LatentDirichletAllocation as lda
from utils.spliting_train_test import get_train_test_set
from joblib import dump
import pandas as pd
import gc


def run_LDA(no_X: bool, fimo: bool, dimens: range, use_test: bool = False,
            doc_topic_prior=None, topic_word_prior=None,
            learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10,
            batch_size=128, evaluate_every=-1, total_samples=1000000.0,
            perp_tol=0.1, mean_change_tol=0.001,
            max_doc_update_iter=100,
            n_jobs=None,
            verbose=0,
            random_state=43):

    X_train, X_test, _, _ = get_train_test_set(no_X=no_X, fimo=fimo)

    for i in dimens:
        lda_model = lda(
            n_components=i, random_state=random_state, n_jobs=n_jobs,
            doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior,
            learning_method=learning_method, learning_decay=learning_decay,
            learning_offset=learning_offset, max_iter=max_iter,
            batch_size=batch_size, evaluate_every=evaluate_every, total_samples=total_samples,
            perp_tol=perp_tol, mean_change_tol=mean_change_tol,
            max_doc_update_iter=max_doc_update_iter,
            verbose=verbose,
            random_state=random_state)
        if not use_test:
            lda_model.fit(X_train)
            dump(lda_model, 'LDA_models/LDA_{}_train_only'.format(i))
        else:
            lda_model.fit(pd.concat([X_train, X_test]))
            dump(lda_model, 'LDA_models/LDA_{}_train_test'.format(i))
        del lda_model
        gc.collect()
