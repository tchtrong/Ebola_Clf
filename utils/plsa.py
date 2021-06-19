from typing import Tuple, Union
from numpy import empty, ndarray, einsum, abs, log, inf, finfo
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
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

    def __init__(self, n_components: int = 100,
                 max_iter: int = 10, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter

    # def __repr__(self) -> str:
    #     title = self.__class__.__name__
    #     header = f'{title}:\n'
    #     divider = '=' * len(title) + '\n'
    #     n_topics = f'Number of topics:     {self.n_components}\n'
    #     n_docs = f'Number of documents:  {self._doc_word.shape[0]}\n'
    #     n_words = f'Number of words:      {self._doc_word.shape[1]}\n'
    #     iterations = f'Number of iterations: {len(kl_divergences)}'
    #     body = n_topics + n_docs + n_words + iterations
    #     return header + divider + body

    # @property
    # def tf_idf(self) -> bool:
    #     """Use inverse document frequency to weigh the document-word counts?"""
    #     return self.__tf_idf

    def fit(self, X: ndarray, y=None,
            eps: float = 1e-5,
            warmup: int = 5):
        """Run EM-style training to find latent topics in documents.

        Expectation-maximization (EM) iterates until either the maximum number
        of iterations is reached or if relative changes of the Kullback-
        Leibler divergence between the actual document-word probability
        and its approximate fall below a certain threshold, whichever
        occurs first.

        Since all quantities are update in-place, calling the ``fit`` method
        again after a successful run (possibly with changed convergence
        criteria) will continue to add more iterations on top of the status
        quo rather than starting all over again from scratch.

        Because a few EM iterations are needed to get things going, you can
        specify an initial `warm-up` period, during which progress in the
        Kullback-Leibler divergence is not tracked, and which does not count
        towards the maximum number of iterations.


        Parameters
        ----------
        eps: float, optional
            The convergence cutoff for relative changes in the Kullback-
            Leibler divergence between the actual document-word probability
            and its approximate. Defaults to 1e-5.
        max_iter: int, optional
            The maximum number of iterations to perform. Defaults to 200.
        warmup: int, optional
            The number of iterations to perform before changes in the
            Kullback-Leibler divergence are tracked for convergence.

        Returns
        -------
        PlsaResult
            Container class for the results of the latent semantic analysis.

        """
        n_docs = X.shape[0]
        n_words = X.shape[1]

        random_state: RandomState = check_random_state(self.random_state)
        joint = empty((self.n_components, n_docs, n_words))
        conditional = random_state.normal(
            size=(self.n_components, n_docs, n_words))
        norm = empty((n_docs, self.n_components))
        doc_given_topic = empty((n_docs, self.n_components))
        topic = empty(self.n_components)
        negative_entropy = self.__negative_entropy(X)
        kl_divergences = []
        eps = abs(float(eps))
        warmup = abs(int(warmup))

        # X = X / X.sum()
        # print(X)

        n_iter = 0
        while n_iter < self.max_iter + warmup:
            word_given_topic = empty((n_docs, self.n_components))
            doc_given_topic = self.__normalize(
                einsum('dw,tdw->dt', X, conditional))[0]
            word_given_topic = self.__normalize(
                einsum('dw,tdw->wt', X, conditional))[0]

            topic = einsum('dw,tdw->t', X, conditional)
            joint = einsum('dt,wt,t->tdw',
                           doc_given_topic,
                           word_given_topic,
                           topic)

            conditional, norm = self.__normalize(joint)
            likelihood = (X * log(norm)).sum().sum()
            new_kl_divergence = negative_entropy - likelihood
            n_iter += 1
            if n_iter > warmup and self.__rel_change(kl_divergences, new_kl_divergence) < eps:
                break
            kl_divergences.append(new_kl_divergence)

        self.components_ = self._invert(word_given_topic, topic)

        return self

    # def _m_step(self, X, y=None) -> None:
    #     """This must be implemented for each specific PLSA flavour."""

    #     raise NotImplementedError

    # def _result(self) -> PlsaResult:
    #     """This must be implemented for each specific PLSA flavour."""
    #     raise NotImplementedError

    # def __e_step(self) -> None:
    #     """The E-step of the EM algorithm is the same for all PLSA flavours.

    #     From the joint probability `p(t, d, w)` of latent topics, documents,
    #     and words, we need to get a new conditional probability `p(t|d, w)`
    #     by dividing the joint by the marginal `p(d, w)`.

    #     """
    #     conditional, norm = self.__normalize(joint)

    # def __random(self, n_docs: int, n_words: int) -> ndarray:
    #     """Randomly initialize the conditional probability p(t|d, w)."""
    #     conditional = rand(self.n_components, n_docs, n_words)
    #     return self.__normalize(conditional)[0]

    # def _norm_sum(self, index_pattern: str) -> ndarray:
    #     """Update individual probability factors in the M-step."""
    #     probability = einsum(index_pattern, self._doc_word, conditional)
    #     return self.__normalize(probability)[0]

    def __normalize(self, array: ndarray) -> Tuple[ndarray, Norm]:
        """Normalize probability without underflow or divide-by-zero errors."""
        array[array < MACHINE_PRECISION] = 0.0
        norm = array.sum(axis=0)
        norm[norm == 0.0] = 1.0
        return array / norm, norm

    def __rel_change(self, kl_divergences, new: float) -> float:
        """Return the relative change in the Kullback-Leibler divergence."""
        if kl_divergences:
            old = kl_divergences[-1]
            return abs((new - old) / new)
        return inf

    def _invert(self, conditional: ndarray, marginal: ndarray) -> ndarray:
        """Perform a Bayesian inversion of a conditional probability."""
        inverted = conditional * marginal
        return self.__normalize(inverted.T)[0]

    def __negative_entropy(self, X: ndarray) -> float:
        """Compute the negative entropy of the original document-word matrix."""
        p = X.copy()
        p[p <= MACHINE_PRECISION] = 1.0
        return (p * log(p)).sum().sum()

    # @staticmethod
    # def __validated(corpus: Corpus, n_topics: int) -> Tuple[Corpus, int]:
    #     n_topics = abs(int(n_topics))
    #     if n_topics < 2:
    #         raise ValueError('There must be at least 2 topics!')
    #     if corpus.n_docs <= n_topics or corpus.n_words <= n_topics:
    #         msg = (f'The number of both, documents (= {corpus.n_docs}) '
    #                f'and words (= {corpus.n_words}), must be greater than'
    #                f' {n_topics}, the number of topics!')
    #         raise ValueError(msg)
    #     return corpus, n_topics
