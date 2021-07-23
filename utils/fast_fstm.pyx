# cython: profile=True

cimport cython
# from cython.parallel import prange
from libc.math cimport log as clog
from libc.math cimport fabs as cabs
cimport scipy.linalg.cython_blas as blas
import numpy as np

cdef extern from "immintrin.h" nogil:
    ctypedef double  __m256d
    ctypedef float __m128d
    const int _CMP_EQ_OQ = 0x00
    const int _CMP_NEQ_OQ =	0x0c
    __m256d _mm256_loadu_pd(__m256d *a) nogil
    __m256d _mm256_add_pd(__m256d m1, __m256d m2) nogil
    __m256d _mm256_mul_pd(__m256d m1, __m256d m2) nogil
    __m128d _mm256_castpd256_pd128(__m256d a) nogil
    void _mm256_storeu_pd(__m256d *a, __m256d b) nogil
    __m128d _mm256_extractf128_pd(__m256d m1, const int offset) nogil
    __m128d _mm_unpackhi_pd(__m128d a, __m128d b) nogil
    double _mm_cvtsd_f64(__m128d a) nogil
    __m128d _mm_add_sd(__m128d a, __m128d b) nogil
    __m128d _mm_add_pd(__m128d a, __m128d b) nogil
    __m256d _mm256_set1_pd(double) nogil
    __m256d _mm256_cmp_pd(__m256d m1, __m256d m2, const int predicate) nogil
    __m256d _mm256_or_pd(__m256d m1, __m256d m2) nogil
    __m256d _mm256_and_pd(__m256d m1, __m256d m2) nogil
    __m256d _mm256_div_pd(__m256d m1, __m256d m2) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline double dot_1d(double[:] a, double[:] b):
    cdef int N = a.shape[0]
    cdef int incr = 1
    return blas.ddot(&N, &a[0], &incr, &b[0], &incr)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void matvec(double[:, :] A, double[:] x, double[:] y):
    cdef int N = A.shape[0]
    cdef int D = A.shape[1]
    cdef int n
    cdef int d
    cdef int d_

    cdef:
        __m256d mA, mx, mx1
        __m128d vlow, vhigh, high64

    d_ = D % 16
    for n in range(N):
        y[n] = 0
        for d in range(d_):
            y[n] += A[n, d] * x[d]
        mx = _mm256_set1_pd(0)
        for d in range(d_, D, 16):
            mA = _mm256_loadu_pd(&A[n, d])
            mx1 = _mm256_loadu_pd(&x[d])
            mx1 = _mm256_mul_pd(mA, mx1)
            mx = _mm256_add_pd(mx, mx1)
            mA = _mm256_loadu_pd(&A[n, d + 4])
            mx1 = _mm256_loadu_pd(&x[d + 4])
            mx1 = _mm256_mul_pd(mA, mx1)
            mx = _mm256_add_pd(mx, mx1)
            mA = _mm256_loadu_pd(&A[n, d + 8])
            mx1 = _mm256_loadu_pd(&x[d + 8])
            mx1 = _mm256_mul_pd(mA, mx1)
            mx = _mm256_add_pd(mx, mx1)
            mA = _mm256_loadu_pd(&A[n, d + 12])
            mx1 = _mm256_loadu_pd(&x[d + 12])
            mx1 = _mm256_mul_pd(mA, mx1)
            mx = _mm256_add_pd(mx, mx1)
        vlow  = _mm256_castpd256_pd128(mx);
        vhigh = _mm256_extractf128_pd(mx, 1); # high 128
        vlow  = _mm_add_pd(vlow, vhigh);     # reduce down to 128
        high64 = _mm_unpackhi_pd(vlow, vlow);
        y[n] += _mm_cvtsd_f64(_mm_add_sd(vlow, high64))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double f_1d(double[:] document, double[:] x, int n_features) nogil:
    cdef double result = 0.0
    for idx in range(n_features):
        if x[idx] > 0:
            result += document[idx] * clog(x[idx])
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void grad_f(double[:] document, double[:] x, double[:] grad, int n_features):
    cdef:
        __m256d zero, m_nz, x_, d_
    zero = _mm256_set1_pd(0)
    cdef int i
    cdef int n = n_features % 16

    for i in range(n):
        if x[i] != 0:
            grad[i] = document[i] * (1.0 / x[i])
        else:
            grad[i] = 0

    for i in range(n, n_features, 16):
        x_ = _mm256_loadu_pd(&x[i])
        d_ = _mm256_loadu_pd(&document[i])
        m_nz = _mm256_cmp_pd(zero, x_, _CMP_NEQ_OQ)
        d_ = _mm256_div_pd(d_, x_)
        d_ = _mm256_and_pd(d_, m_nz)
        _mm256_storeu_pd(&grad[i], d_)

        x_ = _mm256_loadu_pd(&x[i + 4])
        d_ = _mm256_loadu_pd(&document[i + 4])
        m_nz = _mm256_cmp_pd(zero, x_, _CMP_NEQ_OQ)
        d_ = _mm256_div_pd(d_, x_)
        d_ = _mm256_and_pd(d_, m_nz)
        _mm256_storeu_pd(&grad[i + 4], d_)

        x_ = _mm256_loadu_pd(&x[i + 8])
        d_ = _mm256_loadu_pd(&document[i + 8])
        m_nz = _mm256_cmp_pd(zero, x_, _CMP_NEQ_OQ)
        d_ = _mm256_div_pd(d_, x_)
        d_ = _mm256_and_pd(d_, m_nz)
        _mm256_storeu_pd(&grad[i + 8], d_)

        x_ = _mm256_loadu_pd(&x[i + 12])
        d_ = _mm256_loadu_pd(&document[i + 12])
        m_nz = _mm256_cmp_pd(zero, x_, _CMP_NEQ_OQ)
        d_ = _mm256_div_pd(d_, x_)
        d_ = _mm256_and_pd(d_, m_nz)
        _mm256_storeu_pd(&grad[i + 12], d_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double alpha_gradient_search(int alpha_max_iter, int n_features,
                                  double[:] document,
                                  double[:] beta, double[:] x,
                                  double[:] nfv):
    cdef double left = 0
    cdef double right = 1
    cdef double fa = 0
    cdef double fa_old = 0
    cdef double alpha = 0
    cdef double sub = 0
    cdef int alpha_iter
    cdef int idx
    for alpha_iter in range (alpha_max_iter):
        alpha = (left + right) / 2
        for idx in range (n_features):
            sub = beta[idx] - x[idx]
            if sub != 0 or x[idx] != 0:
                nfv[idx] = sub / (x[idx] + sub * alpha)
            else:
                nfv[idx] = beta[idx] - x[idx]
        fa = dot_1d(nfv, document)
        if (cabs(fa) < 1e-10 or cabs(fa - fa_old) < 1e-10):
            break
        if (fa < 0):
            right = alpha
        else:
            left = alpha
        fa_old = fa
    return alpha

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int argmax(double[:] grad_f, double[:, :] beta, double[:] ncv,
                int n_components):
    matvec(beta, grad_f, ncv)
    cdef int x = 0
    cdef double max = ncv[0]
    cdef int i
    for i in range(n_components):
        if ncv[i] > max:
            max = ncv[i]
            x = i
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def e_step(int inf_max_iter, double inf_converge, int alpha_max_iter,
           double[:] document,
           double[:, :] beta,
           double[:] x,
           double[:] theta,
           double[:] ncv,
           double[:] nfv):

    cdef int n_components = beta.shape[0]
    cdef int n_features = beta.shape[1]
    cdef double likelihood_old = 0.0
    cdef double converge = 0.0
    cdef double likelihood = 0.0
    cdef double alpha = 0.0
    cdef int i
    cdef int inf_iter
    cdef int x_iter

    for inf_iter in range(inf_max_iter):
        grad_f(document, x, nfv, n_features)
        i = argmax(nfv, beta, ncv, n_components)
        alpha = alpha_gradient_search(alpha_max_iter, n_features, document, beta[i], x, nfv)
        if alpha == 0:
            break
        for x_iter in range(n_features):
            x[x_iter] = alpha * beta[i, x_iter] + (1 - alpha) * x[x_iter]
        for x_iter in range(n_components):
            theta[x_iter] *= (1 - alpha)
        theta[i] += alpha
        likelihood = f_1d(document, x, n_features)
        if inf_iter > 0:
            converge = (likelihood_old - likelihood) / likelihood_old
        likelihood_old = likelihood
        if converge > 0 and converge < inf_converge:
            break

    return likelihood_old
