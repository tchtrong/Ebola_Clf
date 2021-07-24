# %%

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_random_state
from scipy.optimize import minimize_scalar
from utils.fast_fstm import e_step

EPS = np.finfo(float).eps


class FSTM(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components=10, topic_sparsity=0.1, warm_start=True,
                 em_max_iter=10, em_converge=1e-4,
                 inf_max_iter=100, inf_converge=1e-6,
                 alpha_max_iter=20,
                 random_state=None) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.topic_sparsity = topic_sparsity
        self.warm_start = warm_start
        self.em_max_iter = em_max_iter
        self.em_convergence = em_converge
        self.inf_max_iter = inf_max_iter
        self.inf_converge = inf_converge
        self.alpha_max_iter = alpha_max_iter

    def _check_params(self):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r")
        if self.em_max_iter <= 1:
            raise ValueError("Invalid 'em_max_iter' parameter: %r")
        if self.inf_max_iter <= 1:
            raise ValueError("Invalid 'inf_max_iter' parameter: %r")

    def _init_latent_vars(self, n_features):
        self.random_state_ = check_random_state(self.random_state)

        self.components_: np.ndarray = sparse.random(
            self.n_components, n_features, density=0.1, random_state=self.random_state_, dtype=np.float64)

        normalize(self.components_, norm='l1', copy=False)

        self.components_ = self.components_.toarray()

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        X: np.ndarray = self._validate_data(X, ensure_min_features=2)

        n_samples, n_features = X.shape

        self._init_latent_vars(n_features)
        n_components = self.n_components

        # This is theta in literature
        doc_topic_distr = np.zeros(
            shape=(n_samples, n_components), dtype=np.float64)
        ncv = np.zeros(n_components, dtype=np.float64)
        nfv = np.zeros(n_features, dtype=np.float64)

        converge = 1.0
        likelihood_old = 1.0

        for em_iter in range(0, self.em_max_iter):

            likelihood = 0

            # Calculate B at first run or when warm_start is False
            # to not duplicate the calculation
            if not em_iter > 0 or not self.warm_start:
                # Calculate log(x_j)
                log_x_j = self.components_.copy()
                nz = log_x_j != 0
                log_x_j[nz] = np.log(log_x_j[nz])

            for ind_d in range(n_samples):
                if em_iter > 0 and self.warm_start:
                    fx: np.ndarray = self._f(
                        X[ind_d], doc_topic_distr[ind_d][:, np.newaxis] * (self.components_))
                    r = fx.argmax()
                else:
                    # Calculate f(x) = Sigma(d_j * log(x_j))
                    fx: np.ndarray = log_x_j.dot(X[ind_d])
                    r = fx.argmax()
                    doc_topic_distr[ind_d][r] = 1

                likelihood += e_step(self.inf_max_iter, self.inf_converge, self.alpha_max_iter, X[ind_d],
                                     self.components_, self.components_[
                                         r].copy(), doc_topic_distr[ind_d],
                                     ncv, nfv)

            # M-step
            normalize(doc_topic_distr, norm='l1', copy=False, axis=0)

            self.components_ = doc_topic_distr.transpose().dot(X)

            normalize(self.components_, norm='l1', copy=False)

            converge = (likelihood_old - likelihood) / (likelihood_old)
            likelihood_old = likelihood

            if abs(converge) < self.em_convergence:
                break

        return doc_topic_distr

    def _f(self, document: np.ndarray, x: np.ndarray) -> np.ndarray:
        log_x = x.copy()
        nz = log_x != 0
        log_x[nz] = np.log(log_x[nz])
        return document.dot(log_x.transpose())

    def transform(self, X, y=None):
        X: np.ndarray = self._validate_data(X, ensure_min_features=2)

        n_samples, n_features = X.shape
        n_components = self.n_components

        doc_topic_distr = np.zeros(
            shape=(n_samples, n_components), dtype=np.float64)
        ncv = np.zeros(n_components, dtype=np.float64)
        nfv = np.zeros(n_samples, dtype=np.float64)

        log_x_j = self.components_.copy()
        nz = log_x_j != 0
        log_x_j[nz] = np.log(log_x_j[nz])

        for ind_d in range(n_samples):
            fx: np.ndarray = log_x_j.dot(X[ind_d])
            r = fx.argmax()
            doc_topic_distr[ind_d][r] = 1

            e_step(self.inf_max_iter, self.inf_converge, self.alpha_max_iter, X[ind_d],
                   self.components_, self.components_[
                r].copy(), doc_topic_distr[ind_d],
                ncv, nfv)

        normalize(doc_topic_distr, norm='l1', copy=False, axis=0)
        return doc_topic_distr
