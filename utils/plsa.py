from typing import Tuple, Union

import numpy as np
from numpy import abs, einsum, empty, finfo, inf, log, ndarray
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_random_state

Norm = Union[ndarray, None]
Divisor = Union[int, float, ndarray]
MACHINE_PRECISION = finfo(float).eps


class PLSA (TransformerMixin, BaseEstimator):
    """Base class for all flavours of PLSA algorithms.

    Since the base class for all algorithms is not supposed to ever be
    instantiated directly, it is also not documented. For more information,
    please refer to the docstrings of the individual algorithms.

    """

    def __init__(self, n_components: int = 100, max_iter: int = 20, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: ndarray, y=None):
        n_docs = X.shape[0]
        n_words = X.shape[1]

        random_state: RandomState = check_random_state(self.random_state)
        conditional = random_state.normal(
            size=(self.n_components, n_docs, n_words))
        doc_given_topic = empty((n_docs, self.n_components))
        topic = empty(self.n_components)
        word_given_topic = empty((n_words, self.n_components))

        for _ in range(self.max_iter):
            einsum('dw,tdw->dt', X, conditional, out=doc_given_topic, optimize='greedy')
            normalize(doc_given_topic, norm='l1', axis=0, copy=False)

            einsum('dw,tdw->wt', X, conditional, out=word_given_topic, optimize='greedy')
            normalize(word_given_topic, norm='l1', copy=False)

            topic = einsum('dw,tdw->t', X, conditional, out=topic, optimize='greedy')

            einsum('dt,wt,t->tdw', doc_given_topic,
                   word_given_topic, topic, out=conditional, optimize='greedy')

            for i in range(conditional.shape[0]):
                conditional[i] = conditional[i] / np.sum(conditional[i])

        self.components_ = word_given_topic.T * topic[:, np.newaxis]

        return self

    def transform(self, X, y=None):
        return normalize(X.dot(self.components_.T), norm='l1', axis=0)
