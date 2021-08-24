# %%

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_random_state

EPS = np.finfo(float).eps


class FSTM(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=10, topic_sparsity=0.1, warm_start=True,
                 em_max_iter=10, em_converge=1e-4,
                 inf_max_iter=10, inf_converge=1e-6,
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

        self.components_ = [sparse.random(
            1, n_features, density=0.1, random_state=self.random_state_, dtype=np.float64) for _ in range(self.n_components)]

        self.components_ = sparse.vstack(self.components_)

        normalize(self.components_, norm='l1', copy=False)

        self.components_ = self.components_.toarray()

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        X: np.ndarray = self._validate_data(X, ensure_min_features=2)

        n_samples, n_features = X.shape

        ep_big = EPS / (1 - EPS * n_features)
        ep_small = (1 - EPS * n_features)

        self._init_latent_vars(n_features)
        n_components = self.n_components

        doc_topic_distr = np.empty(
            shape=(n_samples, n_components), dtype=np.float64)

        opt = np.empty(shape=n_features, dtype=np.float64)

        log_x_j: np.ndarray = np.empty_like(self.components_, dtype=np.float64)

        converge = 1.0
        likelihood_old = 1.0

        for em_iter in range(0, self.em_max_iter):

            likelihood = 0

            if not em_iter > 0 or not self.warm_start:
                # log_x_j = np.log(self.components_ + ep_big)
                np.log(self.components_ + ep_big, out=log_x_j)

            for ind_d in range(n_samples):
                llh_old = 1
                cv_inf = 1
                if em_iter > 0 and self.warm_start:
                    doc_topic_distr[ind_d][doc_topic_distr[ind_d]
                                           < 0.001] = 0
                    doc_topic_distr[ind_d] /= doc_topic_distr[ind_d].sum()
                    doc_topic_distr[ind_d].dot(self.components_, out=opt)
                    opt[:] = ep_small * opt + EPS
                else:
                    fx: np.ndarray = log_x_j.dot(X[ind_d])
                    r = fx.argmax()
                    opt[:] = EPS
                    i = self.components_[r] > 0
                    opt[i] += ep_small * self.components_[r][i]
                    doc_topic_distr[ind_d] = 0
                    doc_topic_distr[ind_d][r] = 1
                for _ in range(self.inf_max_iter):
                    i = self.components_.dot(
                        np.multiply(X[ind_d], (1/opt))).argmax()
                    x = ep_small * self.components_[i] - opt + EPS
                    alpha = self._alpha_gradient_search(X[ind_d], x, opt)
                    opt += alpha * x
                    doc_topic_distr[ind_d] *= 1 - alpha
                    doc_topic_distr[ind_d][i] += alpha
                    llh = np.multiply(X[ind_d], np.log(opt)).sum()
                    cv_inf = (llh_old - llh) / llh_old
                    llh_old = llh
                    if abs(cv_inf) < self.inf_converge:
                        break
                likelihood += llh_old

            # M-step
            normalize(doc_topic_distr, norm='l1', copy=False)

            self.components_ = doc_topic_distr.transpose().dot(X)

            normalize(self.components_, norm='l1', copy=False)

            converge = (likelihood_old - likelihood) / (likelihood_old)

            likelihood_old = likelihood

            if abs(converge) < self.em_convergence:
                break

        return doc_topic_distr

    def _alpha_gradient_search(self, doc: np.ndarray, x: np.ndarray, opt: np.ndarray) -> float:
        alpha = 0
        left = 0
        right = 1
        for _ in range(self.alpha_max_iter):
            alpha = (left + right) / 2
            fa = (doc * x / (alpha * x + opt)).sum()
            if (abs(fa) < 1e-10):
                break
            if (fa < 0):
                right = alpha
            else:
                left = alpha
        return alpha

    def transform(self, X, y=None):
        X: np.ndarray = self._validate_data(X, ensure_min_features=2)

        n_samples, n_features = X.shape
        n_components = self.n_components

        ep_big = EPS / (1 - EPS * n_features)
        ep_small = (1 - EPS * n_features)

        doc_topic_distr = np.zeros(
            shape=(n_samples, n_components), dtype=np.float64)

        opt = np.empty(shape=n_features, dtype=np.float64)

        log_x_j: np.ndarray = np.empty_like(self.components_, dtype=np.float64)

        np.log(self.components_ + ep_big, out=log_x_j)

        for ind_d in range(n_samples):
            llh_old = 1
            cv_inf = 1
            fx: np.ndarray = log_x_j.dot(X[ind_d])
            r = fx.argmax()
            opt[:] = EPS
            i = self.components_[r] > 0
            opt[i] += ep_small * self.components_[r][i]
            doc_topic_distr[ind_d] = 0
            doc_topic_distr[ind_d][r] = 1
            for _ in range(self.inf_max_iter):
                i = self.components_.dot(
                    np.multiply(X[ind_d], (1/opt))).argmax()
                x = ep_small * self.components_[i] - opt + EPS
                alpha = self._alpha_gradient_search(X[ind_d], x, opt)
                opt += alpha * x
                doc_topic_distr[ind_d] *= 1 - alpha
                doc_topic_distr[ind_d][i] += alpha
                llh = np.multiply(X[ind_d], np.log(opt)).sum()
                cv_inf = (llh_old - llh) / llh_old
                llh_old = llh
                if abs(cv_inf) < self.inf_converge:
                    break

        normalize(doc_topic_distr, norm='l1', copy=False)

        return doc_topic_distr
