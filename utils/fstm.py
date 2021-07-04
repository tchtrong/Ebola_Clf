# %%

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_random_state
from scipy.optimize import minimize_scalar

EPS = np.finfo(float).eps


class FSTM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, topic_sparsity=0.1, warm_start=True,
                 em_max_iter=100, em_converge=1e-4,
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

        # Self component as sparse matrix
        # presents topic-word distribution matrix
        # This is equivalent to B in literature
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
                # Calculate f(x) = Sigma(d_j * log(x_j))
                # j = ind_d
                # x_j = theta_j * B
                # At first run we don't have theta (all is zero), therefore we have to divide
                # the problem to 2 cases
                if em_iter > 0 and self.warm_start:
                    fx: np.ndarray = self._f(
                        X[ind_d], doc_topic_distr[ind_d].dot(self.components_))
                    r = fx.argmax()
                else:
                    # Calculate f(x) = Sigma(d_j * log(x_j))
                    fx: np.ndarray = log_x_j.dot(X[ind_d])
                    r = fx.argmax()
                    doc_topic_distr[ind_d][r] = 1

                likelihood += self._e_step(X[ind_d],
                                           self.components_, r, doc_topic_distr[ind_d])

            # M-step
            self.components_ = doc_topic_distr.transpose().dot(X)
            normalize(self.components_, norm='l1', copy=False)

            if em_iter > 0:
                converge = (likelihood_old - likelihood) / (likelihood_old)
            likelihood_old = likelihood

            if converge < -0.001:
                converge = 1.0

            if converge < self.em_convergence:
                break

        return doc_topic_distr

    def _f(self, document: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate f(x) in literature: f(x) = sum_{j}(d_j * log(x_j)) = d * log(x).T

        Parameters
        ----------
        document : 1 x n_features matrix

        x : {sparse matrix} n_components x n_features

        Returns
        -------
        1 x n_components matrix
        """

        log_x = x.copy()
        nz = log_x != 0
        log_x[nz] = np.log(log_x[nz])

        return document.dot(log_x.transpose())

    def _grad_f(self, document: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate gradient of f(x) in literature

        Parameters
        ----------
        document : 1 x n_features matrix

        x : {sparse matrix} n_components x n_features

        Returns
        -------
        1 x n_components matrix
        """

        x_ = x.copy()
        nz = x_ != 0
        x_[nz] = 1 / x_[nz]

        return np.multiply(document, x_)

    def _f_alpha(self, alpha: float, document: np.ndarray, x: np.ndarray, beta_i: np.ndarray) -> np.float64:
        """Calculate f(x) in literature: f(x) = sum_{j}(d_j * log(x_j)) = d * log(x).T

        Parameters
        ----------
        document : 1 x n_features matrix

        x : {sparse matrix} n_components x n_features

        Returns
        -------
        1 x n_components matrix or 1 x 1 matrix (scaler)
        """

        new_x = x + (beta_i - x) * alpha

        return (self._f(document, new_x) * -1)

    def _e_step(self, document: np.ndarray, beta: np.ndarray, x_ind: int, theta: np.ndarray) -> float:
        x = beta[x_ind]
        likelihood_old = 0
        converge = 0
        for inf_iter in range(self.inf_max_iter):
            grad = self._grad_f(document, x)
            i = grad.dot(beta.transpose()).argmax()
            alpha = minimize_scalar(
                self._f_alpha, bounds=(0, 1), args=(document, x, beta[i]), method='bounded', options={'maxiter': self.alpha_max_iter})
            x = alpha.x * beta[i] + (1 - alpha.x) * x
            theta *= (1 - alpha.x)
            theta[i] += alpha.x
            likelihood = self._f(document, x)
            if inf_iter > 0:
                converge = (likelihood_old - likelihood)/likelihood_old
            likelihood_old = likelihood
            if converge > 0 and converge < self.inf_converge:
                break
        return likelihood_old

    def transform(self, X, y=None):
        X: np.ndarray = self._validate_data(X, ensure_min_features=2)

        n_samples, n_features = X.shape
        n_components = self.n_components

        doc_topic_distr = np.zeros(
            shape=(n_samples, n_components), dtype=np.float64)

        log_x_j = self.components_.copy()
        nz = log_x_j != 0
        log_x_j[nz] = np.log(log_x_j[nz])

        for ind_d in range(n_samples):
            fx: np.ndarray = log_x_j.dot(X[ind_d])
            r = fx.argmax()
            doc_topic_distr[ind_d][r] = 1

            self._e_step(X[ind_d],
                         self.components_, r, doc_topic_distr[ind_d])

        return doc_topic_distr
