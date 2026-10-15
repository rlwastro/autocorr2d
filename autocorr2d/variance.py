"""Compute variances for autocorrelation pair count arrays

The variance and covariance functions operate on externally
supplied padded 1-D separable factors:

    d(x, y) = f(x) g(y),
    r(x, y) = a(x) b(y).

All probability arrays contain probability mass per pixel. Physical values are
normalized internally. Only NumPy and the scipy.fft modules are needed.

This was created with help from Microsoft Copilot, but was extensively edited
to simplify the code (particularly the handling of padding).  This version only
computes the variance for the DD/RR array; the code for the Landy-Szalay
variance and covariance has been removed.

R. White, 2026 October 14
"""

import numpy as np
import scipy.fft

__all__ = [
    "variances",
]


def variances(
    f_padded,
    g_padded,
    a_padded,
    b_padded,
    support_shape,
    poisson_mean,
    x_lag_min=None,
    x_lag_max=None,
    y_lag_min=None,
    y_lag_max=None,
    rr_min_fraction=1e-10,
    zero_tolerance=0.0,
):
    """Compute DD-only variance maps from padded separable factors.

    Parameters
    ----------
    f_padded : array_like
        Padded x factor of the data mass, ``d(x, y) = f(x) g(y)``.
    g_padded : array_like
        Padded y factor of the data mass.
    a_padded : array_like
        Padded x factor of the selection mass, ``r(x, y) = a(x) b(y)``.
    b_padded : array_like
        Padded y factor of the selection mass.
    support_shape : tuple of int
        Unpadded two-dimensional shape ``(ny, nx)``.
    poisson_mean : float
        Fixed expected Poisson count Lambda.
    x_lag_min, x_lag_max : int or None, optional
        Inclusive requested x-lag bounds. None selects the full bound.
    y_lag_min, y_lag_max : int or None, optional
        Inclusive requested y-lag bounds. None selects the full bound.
    rr_min_fraction : float, optional
        Mask RR values below this fraction of maximum requested RR.
    zero_tolerance : float or None, optional
        Padding-check tolerance. None disables the numerical check.

    Returns
    -------
    variance map over the requested lag region

    Notes
    -----
    The Q terms are computed with one-dimensional separable products. The DD
    population and RR terms factor into x and y autocorrelations. Consequently,
    no two-dimensional variance array is shifted or transformed.
    """
    poisson_mean = float(poisson_mean)

    if not np.isfinite(poisson_mean) or poisson_mean <= 0:
        raise ValueError("poisson_mean must be finite and positive")

    lags, lag_y, lag_x = _lag_rectangle(
        support_shape,
        x_lag_min,
        x_lag_max,
        y_lag_min,
        y_lag_max,
    )
    unique_x_lags = lag_x
    unique_y_lags = lag_y

    ff = _fft_autocorrelation_values(
        f_padded,
        unique_x_lags,
    )
    aa = _fft_autocorrelation_values(
        a_padded,
        unique_x_lags,
    )
    gg = _fft_autocorrelation_values(
        g_padded,
        unique_y_lags,
    )
    bb = _fft_autocorrelation_values(
        b_padded,
        unique_y_lags,
    )

    dd_population = np.outer(gg, ff)
    rr = np.outer(bb, aa)
    q_dd = _separable_q_diagonal(
        f_padded,
        g_padded,
        a_padded,
        b_padded,
        lags,
    ).reshape(rr.shape)
    p_shot_noise = 0.5 * dd_population
    zero_y = np.where(lag_y == 0)[0]
    zero_x = np.where(lag_x == 0)[0]

    # do not divide by rr, and multiply by square of
    # poisson_mean*(poisson_mean-1) to get variance in pairs array
    variance_map = (
        q_dd / poisson_mean
        + 2 * p_shot_noise / poisson_mean**2
    ) * (poisson_mean*(poisson_mean-1))**2
    # this is not strictly necessary, but leave it for now
    valid = rr > rr.max() * rr_min_fraction
    variance_map[~valid] = np.nan
    return variance_map


def _lag_rectangle(
    support_shape,
    x_lag_min,
    x_lag_max,
    y_lag_min,
    y_lag_max,
):
    """Create an inclusive rectangular lag list and its coordinate axes."""
    ny, nx = [int(x) for x in support_shape]
    x_lag_min = -nx + 1 if x_lag_min is None else int(x_lag_min)
    x_lag_max = nx - 1 if x_lag_max is None else int(x_lag_max)
    y_lag_min = -ny + 1 if y_lag_min is None else int(y_lag_min)
    y_lag_max = ny - 1 if y_lag_max is None else int(y_lag_max)

    if (
        x_lag_min > x_lag_max
        or y_lag_min > y_lag_max
        or x_lag_min < -nx + 1
        or x_lag_max > nx - 1
        or y_lag_min < -ny + 1
        or y_lag_max > ny - 1
    ):
        raise ValueError("invalid lag bounds")

    lag_y = np.arange(y_lag_min, y_lag_max + 1)
    lag_x = np.arange(x_lag_min, x_lag_max + 1)
    lags = np.array(
        [(dy, dx) for dy in lag_y for dx in lag_x],
        dtype=int,
    )
    #XXX implement meshgrid version of lags
    #XXX lags2 = np.meshgrid(lag_y, lag_x)
    #XXX something something
    return lags, lag_y, lag_x


def _fft_autocorrelation_values(
    array_padded,
    lags,
):
    """Compute selected one-dimensional autocorrelation values by FFT."""
    transform = scipy.fft.rfft(array_padded, norm="backward")
    correlation = scipy.fft.fftshift(
        scipy.fft.irfft(np.conj(transform) * transform, norm="backward",
                        n=array_padded.shape[0])
    )
    center = array_padded.size // 2
    # return np.array([correlation[center + lag] for lag in lags])
    return correlation[center + lags]


def _separable_q_diagonal(f, g, a, b, lags):
    """Compute all diagonal Q terms with separable one-dimensional sums."""
    values = np.empty(len(lags), dtype=float)

    for index, (dy, dx) in enumerate(lags):
        components = _lag_components(
            f,
            g,
            a,
            b,
            dy,
            dx,
        )
        values[index] = _separable_q_entry(
            f,
            g,
            components,
            components,
        )

    return values


def _lag_components(f, g, a, b, dy, dx):
    """Return separable signed components for one influence function.

    Array is padded, so np.roll (circular shift) is appropriate for shift.
    """
    x_parts = [
        np.roll(f, dx),
        np.roll(f, -dx),
    ]
    y_parts = [
        np.roll(g, dy),
        np.roll(g, -dy),
    ]
    coefficients = [1.0, 1.0]
    return x_parts, y_parts, coefficients


def _separable_q_entry(
    f,
    g,
    first_components,
    second_components,
):
    """Compute one shared-point Q entry using one-dimensional products."""
    first_x, first_y, first_coefficients = first_components
    second_x, second_y, second_coefficients = second_components
    value = 0.0

    for first_index, first_coefficient in enumerate(first_coefficients):
        for second_index, second_coefficient in enumerate(second_coefficients):
            x_value = np.sum(
                f
                * first_x[first_index]
                * second_x[second_index]
            )
            y_value = np.sum(
                g
                * first_y[first_index]
                * second_y[second_index]
            )
            value += (
                first_coefficient
                * second_coefficient
                * x_value
                * y_value
            )

    return value

