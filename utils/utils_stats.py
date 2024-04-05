"""
This file contains some functions related to probability distributions,
mainly those that are available in matlab, R, etc., but not in numpy, scipy.
"""

import random
import warnings
from math import atan2, sqrt
from numbers import Real
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats as ss
from scipy.linalg import solve_triangular
from scipy.spatial import distance as scipy_dist

__all__ = [
    "modulo",
    "autocorr",
    "crosscorr",
    "kmeans2_is_correct",
    "is_outlier",
    "log_multivariate_normal_density",
    "mahalanobis",
    "log_likelihood",
    "likelihood",
    "covariance_ellipse",
    "_eigsorted",
    "rand_student_t",
    "BasicStats",
]


# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    ss.multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        "Using a version of SciPy that does not support the "
        "allow_singular parameter in scipy.stats.multivariate_normal.logpdf().",
        DeprecationWarning,
    )
    _support_singular = False


def modulo(val: Real, dividend: Real, val_range_start: Real = 0) -> Real:
    """Compute the modulo of val by dividend, positive,
    and within interval [val_range_start, val_range_start+abs(dividend)]

    Parameters
    ----------
    val : numbers.Real,
        The value to be computed
    dividend : numbers.Real,
        The divisor.
    val_range_start : numbers.Real, default 0,
        The start of the interval.

    Returns
    -------
    numbers.Real
        val mod dividend, positive,
        and within interval [val_range_start, val_range_start+abs(dividend)].
    """
    _dividend = abs(dividend)
    ret = val - val_range_start - _dividend * int((val - val_range_start) / _dividend)
    return ret + val_range_start if ret >= 0 else _dividend + ret + val_range_start
    # alternatively
    # return (val-val_range_start)%_dividend + val_range_start


def autocorr(x: Union[list, tuple, np.ndarray, pd.Series], normalize: bool = False) -> np.ndarray:
    """Autocorrelation of the time series x

    Parameters
    ----------
    x : `array_like`
        The time series for analysis.
    normalize : bool, default False
        If True, normalize the result.

    Returns
    -------
    numpy.ndarray
        The autocorrelation of the time series x.

    """
    if normalize:
        _x = np.array(x) - np.mean(x)
        result = np.correlate(_x, _x, mode="full")
        result = result[result.size // 2 :]
        result = result / np.sum(np.power(_x, 2))
    else:
        result = np.correlate(x, x, mode="full")[result.size // 2 :]
    return result


def crosscorr(x: pd.Series, y: pd.Series, lag: int = 0, wrap: bool = False) -> float:
    """Lag-N cross correlation.

    Parameters
    ----------
    lag : int, default 0
        Window lag (offset).
    x, y : pandas.Series
        Time series data, of equal length.
    warp : bool, default False
        If True, circular shift is applied to the data.

    Returns:
    --------
    float
        Lag-N cross correlation of the time series x and y.

    """
    shifted_y = y.shift(lag)  # Shifted data filled with NaNs by default
    if wrap:
        shifted_y.iloc[:lag] = y.iloc[-lag:].values
        cc = x.corr(shifted_y)
    else:
        cc = x.corr(shifted_y)
    return cc


def kmeans2_is_correct(data: np.ndarray, centroids: np.ndarray, labels: np.ndarray, verbose: int = 0) -> bool:
    """Check if the result of scipy.cluster.vq.kmeans2 is correct.

    Parameters
    ----------
    data : numpy.ndarray
        The data array.
    centroids : numpy.ndarray
        The centroids computed by kmeans2.
    labels : numpy.ndarray
        The labels.
    verbose : int, default 0
        The verbosity level.

    Returns
    -------
    bool
        True if the result is correct, False otherwise.

    """
    nb_clusters = len(centroids)
    nb_clusters2 = len(set(labels))

    if verbose >= 1:
        print("nb_clusters(len(centroids)) = {0}, nb_clusters2(len(set(labels))) = {1}".format(nb_clusters, nb_clusters2))
        if verbose >= 2:
            print("data =", data)
            print("centroids =", centroids)
            print("labels =", labels)

    if nb_clusters != nb_clusters2:
        return False

    if nb_clusters == 1:
        if np.nanmax(data) / np.nanmin(data) >= 1.5:
            return False
        else:
            return True

    to_check = [lb for lb in range(nb_clusters) if (labels == lb).sum() > 1]
    if verbose >= 1:
        print("to_check =", to_check)
        print("np.sign(data-centroids[lb]) =", [np.sign(data - centroids[lb]) for lb in to_check])
    return all([len(set(np.sign(data - centroids[lb]))) >= 2 for lb in to_check])  # == 2 or == 3


def is_outlier(
    to_check_val: Real, normal_vals: Union[List[int], List[float], Tuple[int], Tuple[float], np.ndarray], verbose: int = 0
) -> bool:
    """Check if `to_check_val` is an outlier in `normal_vals`

    Parameters
    ----------
    to_check_val : numbers.Real
        The value to check.
    normal_vals : Union[List[int],List[float],Tuple[int],Tuple[float],np.ndarray]
        The normal values.
    verbose : int, default 0
        The verbosity level.

    Returns
    -------
    bool
        True if `to_check_val` is an outlier, False otherwise.

    """
    perc75, perc25 = np.percentile(normal_vals, [75, 25])
    iqr = perc75 - perc25
    lower_bound = perc25 - 1.5 * iqr
    upper_bound = perc75 + 1.5 * iqr
    if verbose >= 1:
        print(
            "75 percentile = {0}, 25 percentile = {1}, iqr = {2}, lower_bound = {3}, upper_bound = {4}".format(
                perc75, perc25, iqr, lower_bound, upper_bound
            )
        )
    return not lower_bound <= to_check_val <= upper_bound


def log_multivariate_normal_density(
    X: Union[list, tuple, np.ndarray, pd.Series],
    means: Union[list, tuple, np.ndarray, pd.Series],
    covars: Union[list, tuple, np.ndarray, pd.Series],
    min_covar: float = 1.0e-7,
) -> np.ndarray:
    """Log probability for full covariance matrices.

    Parameters
    ----------
    X : `array_like`
        List of n_features-dimensional data points. Each row corresponds to a single data point,
        of shape ``(n_samples, n_features)``.
    means : `array_like`
        List of n_features-dimensional mean vectors for each component,
        of shape ``(n_components, n_features)``.
    covars : `array_like`
        List of n_features x n_features covariance matrices for each component,
        of shape ``(n_components, n_features, n_features)``.
    min_covar : float, default 1.e-7
        Floor on the diagonal of the covariance matrix to prevent divison by zero.

    Returns
    -------
    log_prob : `array_like`
        Log probabilities of each data point in X, of shape ``(n_samples, n_components)``.

    """
    _X = np.array(X)
    n_samples, n_dim = _X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim))
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (_X - mu).T).T
        log_prob[:, c] = -0.5 * (np.sum(cv_sol**2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def mahalanobis(
    x: Union[list, tuple, np.ndarray, float],
    mean: Union[list, tuple, np.ndarray, float],
    cov: Union[list, tuple, np.ndarray, float],
) -> float:
    """Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`.

    This can be thought as the number of standard deviations x is from the mean,
    i.e. a return value of `3` means that x is `3` std from mean.

    Parameters
    ----------
    x : `array_like` or float
        Input state vector, of shape `(N,)` if `x` is a vector.
    mean : `array_like` or float
        Mean of multivariate Gaussian.
    cov : `array_like` or float
        Covariance of the multivariate Gaussian.

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `x` and `mean`.

    Examples
    --------
    >>> mahalanobis(x=3., mean=3.5, cov=4.**2) # univariate case
    0.125
    >>> mahalanobis(x=3., mean=6, cov=1) # univariate, 3 std away
    3.0
    >>> mahalanobis([1., 2], [1.1, 3.5], [[1., .1],[.1, 13]])
    0.42533327058913922

    """
    _x = scipy_dist._validate_vector(x)
    _mean = scipy_dist._validate_vector(mean)

    if _x.shape != _mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = _x - _mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
    return sqrt(dist)


def log_likelihood(
    z: Union[np.ndarray, float, int], x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray
) -> Union[np.ndarray, float, int]:
    """Computes log-likelihood of the measurement z given the Gaussian posterior (x, P)
    using measurement function H and measurement covariance error R.

    Parameters
    ----------
    z : `array_like` or float
        Measurement vector.
    x : `array_like`
        State vector.
    P : `array_like`
        Covariance matrix of the state.
    H : `array_like`
        Measurement function.
    R : `array_like`
        Covariance matrix of the measurement.

    Returns
    -------
    log_likelihood : double
        The log-likelihood of the measurement `z` given the Gaussian posterior `(x, P)`.

    """
    S = np.dot(H, np.dot(P, H.T)) + R
    return ss.multivariate_normal.logpdf(z, np.dot(H, x), S)


def likelihood(
    z: Union[np.ndarray, float, int], x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray
) -> Union[np.ndarray, float, int]:
    """Computes likelihood of the measurement z given the Gaussian posterior (x, P)
    using measurement function H and measurement covariance error R.

    Parameters
    ----------
    z : `array_like` or float
        Measurement vector.
    x : `array_like`
        State vector.
    P : `array_like`
        Covariance matrix of the state.
    H : `array_like`
        Measurement function
    R : `array_like`
        Covariance matrix of the measurement.

    Returns
    -------
    likelihood : double
        The likelihood of the measurement `z` given the Gaussian posterior `(x, P)`.

    """
    return np.exp(log_likelihood(z, x, P, H, R))


def covariance_ellipse(P: np.ndarray, deviations: int = 1) -> Tuple[float, float, float]:
    """Computes a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------
    P : numpy.ndarray
       Covariance matrix, of shape ``(2,2)``.
    deviations : int, default 1
       Number of standard deviations.

    Returns
    -------
    angle_radians : float
        The angle of rotation of the ellipse in radians.
    width_radius : float
        The radius of the ellipse in the x-direction (width).
    height_radius : float
        The radius of the ellipse in the y-direction (height).

    """

    U, s, _ = linalg.svd(P)
    orientation = atan2(U[1, 0], U[0, 0])
    width = deviations * sqrt(s[0])
    height = deviations * sqrt(s[1])

    if height > width:
        raise ValueError("width must be greater than height")

    return (orientation, width, height)


def _eigsorted(cov: np.ndarray, asc: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.

    Parameters
    ----------
    cov : numpy.ndarray
        The covariance matrix.
    asc : bool, default=True
        Determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)

    Returns
    -------
    eigval : numpy.ndarray
        1D array of eigenvalues of covariance ordered largest to smallest
    eigvec : numpy.ndarray
        2D array of eigenvectors of covariance matrix ordered to match `eigval` ordering,
        i.e eigvec[:, 0] is the rotation vector for eigval[0].

    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]


def rand_student_t(df: Real, mu: Real = 0, std: Real = 1) -> Real:
    """Sample a random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.

    Parameters
    ----------
    df : numbers.Real
        The degrees of freedom.
    mu : numbers.Real, default 0
        The mean.
    std : numbers.Real, default 1
        The standard deviation.

    Returns
    -------
    numbers.Real
        The random number distributed by student's t distribution.

    """

    x = random.gauss(0, std)
    y = 2.0 * random.gammavariate(0.5 * df, 2.0)
    return x / (sqrt(y / df)) + mu


class BasicStats(object):
    """Basic statistics of a sequence.

    Parameters
    ----------
    seq : Union[list, tuple, np.ndarray, pd.Series]
        The sequence for analysis.

    Attributes
    ----------
    mean : float
        The mean of the sequence.
    variance : float
        The variance of the sequence.
    std : float
        The standard deviation of the sequence.
    interquartile_range : float
        The interquartile range of the sequence.
    skewness : float
        The skewness of the sequence.
    kurtosis : float
        The kurtosis of the sequence.

    """

    def __init__(self, seq: Union[list, tuple, np.ndarray, pd.Series]):
        self.seq = seq

    @property
    def mean(self):
        return np.mean(self.seq)

    @property
    def variance(self):
        return np.var(self.seq)

    @property
    def std(self):
        return np.std(self.seq)

    @property
    def interquartile_range(self):
        q_1, q_3 = np.percentile(self.seq, [25, 75])
        return abs(q_1 - q_3)

    @property
    def skewness(self):
        return ss.skew(self.seq)

    @property
    def kurtosis(self):
        return ss.kurtosis(self.seq)
