"""Padded-circular two-dimensional Landy-Szalay calculations.

Compute Landy-Szalay autocorrelation for a 2-D point distribution
Compute variance under the assumption that the distribution is separable,
but do not restrict the computed 2-D correlation itself to be separable.
That should allow small non-separable components such as light echoes to
be detected.

This version uses padded arrays that are treated as periodic.  That improves the
handling of the FFT results.

All empirical and model arrays are supplied on one shared zero-padded domain.
The module never pads inside the correlation, variance, or covariance routines.
This makes it convenient to reuse arrays padded by external code.

The original nonzero image has shape ``(ny, nx)``. If covariance or pair maps
are needed only for lags satisfying ``abs(hy) <= Hy`` and ``abs(hx) <= Hx``, a
sufficient circular-domain shape is

    padded_ny >= ny + Hy
    padded_nx >= nx + Hx

when the original arrays occupy ``[:ny, :nx]`` and the remaining samples are
zero. For the complete linear-correlation range, use

    padded_ny = 2*ny - 1
    padded_nx = 2*nx - 1.

Circular FFTs and ``numpy.roll`` on this padded domain reproduce zero-extended
linear operations over the requested lag range. They do not impose a periodic
boundary on the original physical image.

R. White / Microsoft Copilot, 2026 October 2
"""

# TODO
# create interface that allows passing improved sampling from my autocorr2d module
# including window fuction
# simplify window function to be a list of regions with zero selection window
# replace histogram2d with bincount2d
# add padding for FFTs
# use real FFTs

import numpy as np


# -----------------------------------------------------------------------------
# Padding sizes and padding helpers
# -----------------------------------------------------------------------------


def required_padding_shape(original_shape, max_lag=None, full_range=False):
    """Return the minimum shared padded shape.

    Parameters
    ----------
    original_shape : sequence of int
        Original unpadded image shape ``(ny, nx)``.
    max_lag : sequence of int, optional
        Maximum requested absolute lag ``(Hy, Hx)``. The required padded shape
        is ``(ny + Hy, nx + Hx)``. This argument is required unless
        ``full_range`` is true.
    full_range : bool, optional
        If true, return ``(2*ny-1, 2*nx-1)``, which supports every linear lag
        of the original image. The default is false.

    Returns
    -------
    tuple
        Required padded shape ``(padded_ny, padded_nx)``.
    """
    ny, nx = int(original_shape[0]), int(original_shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError("original_shape entries must be positive")

    if full_range:
        return 2 * ny - 1, 2 * nx - 1

    if max_lag is None:
        raise ValueError("max_lag is required when full_range is false")

    Hy, Hx = abs(int(max_lag[0])), abs(int(max_lag[1]))
    if Hy >= ny or Hx >= nx:
        raise ValueError("max_lag must be smaller than original_shape")

    return ny + Hy, nx + Hx


def padding_shape_from_lags(original_shape, lags):
    """Return the minimum shared padded shape for a collection of lags.

    Parameters
    ----------
    original_shape : sequence of int
        Original unpadded image shape ``(ny, nx)``.
    lags : sequence
        Sequence of integer vector lags ``(hy, hx)``.

    Returns
    -------
    tuple
        Required padded shape ``(padded_ny, padded_nx)``.
    """
    lags = list(lags)
    if not lags:
        raise ValueError("at least one lag is required")

    Hy = max(abs(int(lag[0])) for lag in lags)
    Hx = max(abs(int(lag[1])) for lag in lags)
    return required_padding_shape(original_shape, (Hy, Hx))


def pad_spatial_array(array, padded_shape):
    """Place a two-dimensional array in the upper-left of a zero-filled domain.

    Parameters
    ----------
    array : array_like
        Original two-dimensional count, mass, or selection array.
    padded_shape : sequence of int
        Target padded shape ``(padded_ny, padded_nx)``.

    Returns
    -------
    numpy.ndarray
        Floating-point padded array. The original values occupy
        ``[:ny, :nx]`` and all remaining entries are zero.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("array must be two-dimensional")

    padded_ny, padded_nx = int(padded_shape[0]), int(padded_shape[1])
    ny, nx = array.shape
    if padded_ny < ny or padded_nx < nx:
        raise ValueError("padded_shape must not be smaller than array.shape")

    output = np.zeros((padded_ny, padded_nx), dtype=array.dtype)
    output[:ny, :nx] = array
    return output


def pad_factor(factor, padded_size):
    """Append zeros to a one-dimensional factor.

    Parameters
    ----------
    factor : array_like
        Original one-dimensional probability or selection factor.
    padded_size : int
        Target padded length.

    Returns
    -------
    numpy.ndarray
        Floating-point padded factor with the original values first and zeros
        in the remaining entries.
    """
    factor = np.asarray(factor, dtype=np.float64)
    padded_size = int(padded_size)

    if factor.ndim != 1:
        raise ValueError("factor must be one-dimensional")
    if padded_size < factor.size:
        raise ValueError("padded_size must not be smaller than factor length")

    output = np.zeros(padded_size, dtype=np.float64)
    output[:factor.size] = factor
    return output


def pad_all_arrays(counts, selection_mass, F, G, A, B, padded_shape):
    """Pad empirical arrays and separable factors onto one shared domain.

    Parameters
    ----------
    counts : array_like
        Original two-dimensional count image with shape ``(ny, nx)``.
    selection_mass : array_like
        Original two-dimensional selection probability mass with the same
        shape as ``counts``.
    F, G : array_like
        Original probability factors, with lengths ``nx`` and ``ny``.
    A, B : array_like
        Original selection factors, with lengths ``nx`` and ``ny``.
    padded_shape : sequence of int
        Shared target shape ``(padded_ny, padded_nx)``.

    Returns
    -------
    dict
        Dictionary containing padded ``counts``, ``selection_mass``, ``F``,
        ``G``, ``A``, and ``B``, plus ``original_shape`` and ``padded_shape``.
    """
    counts = np.asarray(counts)
    selection_mass = np.asarray(selection_mass, dtype=np.float64)

    if counts.ndim != 2 or counts.shape != selection_mass.shape:
        raise ValueError("counts and selection_mass must have equal 2D shapes")

    ny, nx = counts.shape
    F = _probability_vector(F, "F")
    G = _probability_vector(G, "G")
    A = _probability_vector(A, "A")
    B = _probability_vector(B, "B")

    if F.size != nx or A.size != nx:
        raise ValueError("F and A lengths must equal counts.shape[1]")
    if G.size != ny or B.size != ny:
        raise ValueError("G and B lengths must equal counts.shape[0]")

    padded_ny, padded_nx = int(padded_shape[0]), int(padded_shape[1])

    return {
        "counts": pad_spatial_array(counts, padded_shape),
        "selection_mass": pad_spatial_array(selection_mass, padded_shape),
        "F": pad_factor(F, padded_nx),
        "G": pad_factor(G, padded_ny),
        "A": pad_factor(A, padded_nx),
        "B": pad_factor(B, padded_ny),
        "original_shape": (ny, nx),
        "padded_shape": (padded_ny, padded_nx),
    }


# -----------------------------------------------------------------------------
# Validation and lag axes
# -----------------------------------------------------------------------------


def _probability_vector(values, name):
    """Return a validated normalized one-dimensional probability vector."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(name + " must be one-dimensional")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(name + " must contain finite nonnegative values")
    if not np.isclose(np.sum(values), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(name + " must sum to one")
    return values.copy()


def _check_padded_probability_factor(values, name):
    """Validate a padded factor whose zero padding preserves unit mass."""
    return _probability_vector(values, name)


def circular_lag_axis(size):
    """Return centered signed lags for a circular array.

    Parameters
    ----------
    size : int
        Length of the circular axis. Odd and even lengths are supported.

    Returns
    -------
    numpy.ndarray
        Integer lags in the order corresponding to ``numpy.fft.fftshift``.
    """
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive")

    return np.rint(
        np.fft.fftshift(np.fft.fftfreq(size) * size)
    ).astype(np.int64)


def extract_lag_window(array, all_y_lags, all_x_lags, y_lags, x_lags):
    """Extract a requested rectangular lag window.

    Parameters
    ----------
    array : array_like
        Full padded-size centered-lag array.
    all_y_lags, all_x_lags : array_like
        Full lag axes corresponding to ``array``.
    y_lags, x_lags : array_like
        Requested signed lag values.

    Returns
    -------
    numpy.ndarray
        Requested lag window with shape ``(len(y_lags), len(x_lags))``.
    """
    array = np.asarray(array)
    all_y_lags = np.asarray(all_y_lags, dtype=np.int64)
    all_x_lags = np.asarray(all_x_lags, dtype=np.int64)
    y_lags = np.asarray(y_lags, dtype=np.int64)
    x_lags = np.asarray(x_lags, dtype=np.int64)

    y_lookup = {int(value): index for index, value in enumerate(all_y_lags)}
    x_lookup = {int(value): index for index, value in enumerate(all_x_lags)}

    try:
        y_indices = [y_lookup[int(value)] for value in y_lags]
        x_indices = [x_lookup[int(value)] for value in x_lags]
    except KeyError as error:
        raise ValueError("a requested lag is unavailable") from error

    return array[np.ix_(y_indices, x_indices)]


# -----------------------------------------------------------------------------
# Empirical pair maps from already padded arrays
# -----------------------------------------------------------------------------


def circular_correlation_fft_2d(u_padded, v_padded):
    """Compute a circular two-dimensional correlation without repadding.

    Parameters
    ----------
    u_padded, v_padded : array_like
        Equal-shape two-dimensional arrays already embedded in the shared
        zero-padded circular domain.

    Returns
    -------
    numpy.ndarray
        Full padded-size correlation ``sum_a u[a] v[a+h]`` in centered lag
        order. The output shape equals the input padded shape.
    """
    u_padded = np.asarray(u_padded, dtype=np.float64)
    v_padded = np.asarray(v_padded, dtype=np.float64)

    if u_padded.ndim != 2 or u_padded.shape != v_padded.shape:
        raise ValueError("padded arrays must have equal 2D shapes")

    U = np.fft.fft2(u_padded)
    V = np.fft.fft2(v_padded)
    correlation = np.fft.ifft2(np.conj(U) * V).real
    return np.fft.fftshift(correlation)


def empirical_pair_maps(
    counts_padded,
    selection_mass_padded,
    rr_min_fraction=1.0e-10,
):
    """Compute empirical pair maps from already padded spatial arrays.

    Parameters
    ----------
    counts_padded : array_like
        Two-dimensional count image already embedded in the shared padded
        domain. Padding entries must be zero.
    selection_mass_padded : array_like
        Deterministic selection probability mass on the same padded domain.
        Padding entries must be zero and the complete array must sum to one.
    rr_min_fraction : float, optional
        Lags with ``RR`` below this fraction of the maximum ``RR`` are marked
        invalid. The default is ``1e-10``.

    Returns
    -------
    dict
        Full padded-size centered-lag arrays ``DD``, ``DR``, ``RD``,
        ``DR_symmetric``, ``RR``, ``numerator``, ``xi``, and ``valid``. The
        dictionary also contains ``N``, ``x_lags``, and ``y_lags``.
    """
    counts_padded = np.asarray(counts_padded)
    selection_mass_padded = np.asarray(
        selection_mass_padded,
        dtype=np.float64,
    )

    if counts_padded.ndim != 2 or counts_padded.shape != selection_mass_padded.shape:
        raise ValueError("padded counts and selection arrays must have equal shapes")
    if np.any(counts_padded < 0):
        raise ValueError("counts_padded must be nonnegative")
    if np.any(selection_mass_padded < 0.0):
        raise ValueError("selection_mass_padded must be nonnegative")
    if not np.isclose(np.sum(selection_mass_padded), 1.0):
        raise ValueError("selection_mass_padded must sum to one")

    N = int(np.sum(counts_padded))
    if N < 2:
        raise ValueError("at least two observed points are required")

    data_mass_padded = counts_padded / float(N)
    dd_raw = circular_correlation_fft_2d(data_mass_padded, data_mass_padded)
    dr = circular_correlation_fft_2d(data_mass_padded, selection_mass_padded)
    rd = circular_correlation_fft_2d(selection_mass_padded, data_mass_padded)
    rr = circular_correlation_fft_2d(selection_mass_padded, selection_mass_padded)

    dd = (float(N) / float(N - 1)) * dd_raw

    # In centered lag order, zero lag is located by the circular lag axes.
    y_lags = circular_lag_axis(dd.shape[0])
    x_lags = circular_lag_axis(dd.shape[1])
    zero_y = int(np.where(y_lags == 0)[0][0])
    zero_x = int(np.where(x_lags == 0)[0][0])
    dd[zero_y, zero_x] -= 1.0 / float(N - 1)

    numerator = dd - dr - rd + rr
    valid = rr > rr_min_fraction * float(np.max(rr))
    xi = np.full_like(rr, np.nan)
    xi[valid] = numerator[valid] / rr[valid]

    return {
        "N": N,
        "y_lags": y_lags,
        "x_lags": x_lags,
        "DD": dd,
        "DR": dr,
        "RD": rd,
        "DR_symmetric": 0.5 * (dr + rd),
        "RR": rr,
        "numerator": numerator,
        "xi": xi,
        "valid": valid,
    }


# -----------------------------------------------------------------------------
# Padded separable model pair maps
# -----------------------------------------------------------------------------


def circular_correlation_fft_1d(u_padded, v_padded):
    """Compute a circular one-dimensional correlation without repadding.

    Parameters
    ----------
    u_padded, v_padded : array_like
        Equal-length one-dimensional arrays on the shared padded domain.

    Returns
    -------
    numpy.ndarray
        Full padded-length correlation in native circular FFT order.
    """
    u_padded = np.asarray(u_padded, dtype=np.float64)
    v_padded = np.asarray(v_padded, dtype=np.float64)

    if u_padded.ndim != 1 or u_padded.shape != v_padded.shape:
        raise ValueError("padded factors must have equal one-dimensional shapes")

    return np.fft.ifft(
        np.conj(np.fft.fft(u_padded)) * np.fft.fft(v_padded)
    ).real


def model_pair_maps(F_padded, G_padded, A_padded, B_padded):
    """Compute separable model pair maps from already padded factors.

    Parameters
    ----------
    F_padded, G_padded : array_like
        Padded x- and y-axis probability factors.
    A_padded, B_padded : array_like
        Padded x- and y-axis selection factors. All factors must preserve unit
        mass after padding.

    Returns
    -------
    dict
        Full padded-size centered-lag model maps ``DD``, ``DR``, ``RD``,
        ``DR_symmetric``, ``RR``, ``numerator``, and ``xi``, plus lag axes.
    """
    F = _check_padded_probability_factor(F_padded, "F_padded")
    G = _check_padded_probability_factor(G_padded, "G_padded")
    A = _check_padded_probability_factor(A_padded, "A_padded")
    B = _check_padded_probability_factor(B_padded, "B_padded")

    ff = circular_correlation_fft_1d(F, F)
    fa = circular_correlation_fft_1d(F, A)
    af = circular_correlation_fft_1d(A, F)
    aa = circular_correlation_fft_1d(A, A)
    gg = circular_correlation_fft_1d(G, G)
    gb = circular_correlation_fft_1d(G, B)
    bg = circular_correlation_fft_1d(B, G)
    bb = circular_correlation_fft_1d(B, B)

    dd = np.fft.fftshift(np.outer(gg, ff))
    dr = np.fft.fftshift(np.outer(gb, fa))
    rd = np.fft.fftshift(np.outer(bg, af))
    rr = np.fft.fftshift(np.outer(bb, aa))

    numerator = dd - dr - rd + rr
    xi = np.full_like(rr, np.nan)
    valid = rr > 0.0
    xi[valid] = numerator[valid] / rr[valid]

    return {
        "y_lags": circular_lag_axis(dd.shape[0]),
        "x_lags": circular_lag_axis(dd.shape[1]),
        "DD": dd,
        "DR": dr,
        "RD": rd,
        "DR_symmetric": 0.5 * (dr + rd),
        "RR": rr,
        "numerator": numerator,
        "xi": xi,
    }


# -----------------------------------------------------------------------------
# Low-rank covariance operations on already padded factors
# -----------------------------------------------------------------------------


def _validate_padded_factors(F, G, A, B):
    """Validate and return padded probability and selection factors."""
    F = _check_padded_probability_factor(F, "F_padded")
    G = _check_padded_probability_factor(G, "G_padded")
    A = _check_padded_probability_factor(A, "A_padded")
    B = _check_padded_probability_factor(B, "B_padded")
    return F, G, A, B


def _lag_tuple(lag):
    """Return a lag as ``(hy, hx)``."""
    if len(lag) != 2:
        raise ValueError("each lag must contain exactly (hy, hx)")
    return int(lag[0]), int(lag[1])


def _roll_factor(factor, lag):
    """Return ``factor[i+lag]`` using circular padded-domain indexing."""
    return np.roll(factor, shift=-int(lag))


def _probability_terms(F, G):
    """Return the rank-one representation of p."""
    return [(1.0, G, F)]


def _selection_terms(A, B):
    """Return the rank-one representation of s."""
    return [(1.0, B, A)]


def _apply_B(terms, lag):
    """Apply the symmetric B_h operator to low-rank terms."""
    hy, hx = lag
    output = []

    for coefficient, y_factor, x_factor in terms:
        output.append(
            (
                0.5 * coefficient,
                _roll_factor(y_factor, hy),
                _roll_factor(x_factor, hx),
            )
        )
        output.append(
            (
                0.5 * coefficient,
                _roll_factor(y_factor, -hy),
                _roll_factor(x_factor, -hx),
            )
        )

    return output


def _inner(left_terms, right_terms):
    """Return the full padded-domain inner product of low-rank fields."""
    total = 0.0

    for c_left, y_left, x_left in left_terms:
        for c_right, y_right, x_right in right_terms:
            total += (
                c_left
                * c_right
                * float(np.dot(y_left, y_right))
                * float(np.dot(x_left, x_right))
            )

    return total


def _weighted_inner_p(left_terms, right_terms, F, G):
    """Return ``sum_a p[a] left[a] right[a]`` on the padded domain."""
    total = 0.0

    for c_left, y_left, x_left in left_terms:
        for c_right, y_right, x_right in right_terms:
            y_value = np.sum(G * y_left * y_right, dtype=np.float64)
            x_value = np.sum(F * x_left * x_right, dtype=np.float64)
            total += c_left * c_right * float(y_value) * float(x_value)

    return total


def _multinomial_bilinear(left_terms, right_terms, F, G, N):
    """Return ``left^T C right`` for ``C=[diag(p)-pp^T]/N``."""
    p_terms = _probability_terms(F, G)
    weighted = _weighted_inner_p(left_terms, right_terms, F, G)
    left_mean = _inner(p_terms, left_terms)
    right_mean = _inner(p_terms, right_terms)
    return (weighted - left_mean * right_mean) / float(N)


def _multiply_by_p(terms, F, G):
    """Multiply low-rank terms pointwise by p."""
    return [
        (coefficient, y_factor * G, x_factor * F)
        for coefficient, y_factor, x_factor in terms
    ]


def _p_correlation(F, G, lag):
    """Return the padded circular correlation ``sum_a p[a] p[a+h]``."""
    hy, hx = lag
    x_value = np.dot(F, _roll_factor(F, hx))
    y_value = np.dot(G, _roll_factor(G, hy))
    return float(x_value * y_value)


def _trace_term(h, k, F, G, N):
    """Return ``tr(B_h C B_k C)`` on the padded circular domain."""
    hy, hx = h
    ky, kx = k
    padded_ny = G.size
    padded_nx = F.size
    p_terms = _probability_terms(F, G)
    t0 = 0.0

    for sign_h in (-1, 1):
        for sign_k in (-1, 1):
            combined_y = sign_h * hy + sign_k * ky
            combined_x = sign_h * hx + sign_k * kx

            if combined_y % padded_ny == 0 and combined_x % padded_nx == 0:
                t0 += 0.25 * _p_correlation(
                    F,
                    G,
                    (sign_h * hy, sign_h * hx),
                )

    Bh_p = _apply_B(p_terms, h)
    Bk_p = _apply_B(p_terms, k)

    term_hk = _inner(
        p_terms,
        _apply_B(_multiply_by_p(Bk_p, F, G), h),
    )
    term_kh = _inner(
        p_terms,
        _apply_B(_multiply_by_p(Bh_p, F, G), k),
    )

    mean_h = _inner(p_terms, Bh_p)
    mean_k = _inner(p_terms, Bk_p)
    n_float = float(N)

    return (
        t0 - term_hk - term_kh + mean_h * mean_k
    ) / (n_float * n_float)


def _rr_for_lag(A, B, lag):
    """Return ``RR_h = s^T B_h s`` on the padded domain."""
    s_terms = _selection_terms(A, B)
    return _inner(s_terms, _apply_B(s_terms, lag))


# -----------------------------------------------------------------------------
# Public variance and covariance functions using padded factors directly
# -----------------------------------------------------------------------------


def variance_maps(
    F_padded,
    G_padded,
    A_padded,
    B_padded,
    N,
    x_lags,
    y_lags,
    *,
    rr_min_fraction=1.0e-10,
    factorial_dd_normalization=True,
    RR=None,
):
    """Compute Landy-Szalay and DD-only variance maps.

    Parameters
    ----------
    F_padded, G_padded : array_like
        Already padded x- and y-axis probability factors.
    A_padded, B_padded : array_like
        Already padded x- and y-axis selection factors. All factors must use
        the same padded axis lengths as the empirical spatial arrays.
    N : int
        Number of observed data points.
    x_lags, y_lags : array_like
        Requested signed integer lag axes. The shared padded shape must satisfy
        ``padded_nx >= original_nx + max(abs(x_lags))`` and similarly for y.
    rr_min_fraction : float, optional
        Lags with ``RR`` below this fraction of the maximum requested ``RR``
        are marked invalid. The default is ``1e-10``.
    factorial_dd_normalization : bool, optional
        If true, apply the exact distinct-pair factor ``N/(N-1)`` to every
        covariance term containing DD. The default is true.
    RR : array_like
        2D array with subset of full RR array covering the lags region.
        If omitted, the RR value is computed directly for each lag.

    Returns
    -------
    dict
        Dictionary containing ``RR``, ``variance_LS``, ``sigma_LS``,
        ``variance_DD_only``, ``sigma_DD_only``, validity and lag arrays, and
        numerator-level covariance contributions.
    """
    F, G, A, B = _validate_padded_factors(
        F_padded,
        G_padded,
        A_padded,
        B_padded,
    )

    N = int(N)
    if N < 2:
        raise ValueError("N must be at least two")

    x_lags = np.asarray(x_lags, dtype=np.int64)
    y_lags = np.asarray(y_lags, dtype=np.int64)
    shape = (y_lags.size, x_lags.size)

    if RR is None:
        rr = np.empty(shape, dtype=np.float64)
        for y_index, hy in enumerate(y_lags):
            for x_index, hx in enumerate(x_lags):
                lag = (int(hy), int(hx))
                rr[y_index, x_index] = _rr_for_lag(A, B, lag)
    else:
        if RR.shape != shape:
            raise ValueError("RR must match shape defined by x_lags & y_lags arrays")
        rr = RR

    ls_numerator_variance = np.empty(shape, dtype=np.float64)
    dd_numerator_variance = np.empty(shape, dtype=np.float64)
    dd_dd_piece = np.empty(shape, dtype=np.float64)
    dd_dr_piece = np.empty(shape, dtype=np.float64)
    dr_dr_piece = np.empty(shape, dtype=np.float64)

    p_terms = _probability_terms(F, G)
    s_terms = _selection_terms(A, B)

    if factorial_dd_normalization:
        alpha = float(N) / float(N - 1)
    else:
        alpha = 1.0

    for y_index, hy in enumerate(y_lags):
        for x_index, hx in enumerate(x_lags):
            lag = (int(hy), int(hx))
            Bp = _apply_B(p_terms, lag)
            Bs = _apply_B(s_terms, lag)
            trace = _trace_term(lag, lag, F, G, N)

            covariance_dd_dd = (
                2.0 * trace
                + 4.0 * _multinomial_bilinear(Bp, Bp, F, G, N)
            )
            covariance_dd_dr = 2.0 * _multinomial_bilinear(
                Bp, Bs, F, G, N
            )
            covariance_dr_dd = 2.0 * _multinomial_bilinear(
                Bs, Bp, F, G, N
            )
            covariance_dr_dr = _multinomial_bilinear(
                Bs, Bs, F, G, N
            )

            scaled_dd_dd = alpha * alpha * covariance_dd_dd
            scaled_cross = -2.0 * alpha * (
                covariance_dd_dr + covariance_dr_dd
            )
            scaled_dr_dr = 4.0 * covariance_dr_dr

            dd_dd_piece[y_index, x_index] = scaled_dd_dd
            dd_dr_piece[y_index, x_index] = scaled_cross
            dr_dr_piece[y_index, x_index] = scaled_dr_dr
            dd_numerator_variance[y_index, x_index] = scaled_dd_dd
            ls_numerator_variance[y_index, x_index] = (
                scaled_dd_dd + scaled_cross + scaled_dr_dr
            )

    threshold = rr_min_fraction * float(np.max(rr))
    valid = rr > threshold
    rr_squared = np.square(rr)

    variance_ls = np.full(shape, np.nan, dtype=np.float64)
    variance_dd = np.full(shape, np.nan, dtype=np.float64)
    variance_ls[valid] = ls_numerator_variance[valid] / rr_squared[valid]
    variance_dd[valid] = dd_numerator_variance[valid] / rr_squared[valid]

    return {
        "x_lags": x_lags,
        "y_lags": y_lags,
        "RR": rr,
        "valid": valid,
        "variance_LS": variance_ls,
        "sigma_LS": np.sqrt(np.maximum(variance_ls, 0.0)),
        "variance_DD_only": variance_dd,
        "sigma_DD_only": np.sqrt(np.maximum(variance_dd, 0.0)),
        "LS_numerator_variance": ls_numerator_variance,
        "DD_numerator_variance": dd_numerator_variance,
        "DD_DD_numerator_piece": dd_dd_piece,
        "DD_DR_numerator_piece": dd_dr_piece,
        "DR_DR_numerator_piece": dr_dr_piece,
    }


def covariance_selected_lags(
    F_padded,
    G_padded,
    A_padded,
    B_padded,
    N,
    lags,
    *,
    RR=None,
    factorial_dd_normalization=True,
):
    """Compute complete covariance matrices for selected vector lags.

    Parameters
    ----------
    F_padded, G_padded : array_like
        Already padded x- and y-axis probability factors.
    A_padded, B_padded : array_like
        Already padded x- and y-axis selection factors.
    N : int
        Number of observed data points.
    lags : sequence
        Sequence of signed integer vector lags ``(hy, hx)``. The shared padded
        shape must be large enough for the maximum absolute lag.
    RR : array_like
        Values from full RR array for each lag. If omitted,
        the RR value is computed directly for each lag.
    factorial_dd_normalization : bool, optional
        If true, apply the exact distinct-pair factor ``N/(N-1)`` to every
        covariance term containing DD. The default is true.

    Returns
    -------
    dict
        Dictionary containing ``RR``, component covariance matrices,
        normalized ``cov_LS`` and ``cov_DD_only``, and their diagonal variance
        arrays.
    """
    F, G, A, B = _validate_padded_factors(
        F_padded,
        G_padded,
        A_padded,
        B_padded,
    )

    N = int(N)
    if N < 2:
        raise ValueError("N must be at least two")

    lags = [_lag_tuple(lag) for lag in lags]
    if not lags:
        raise ValueError("at least one lag is required")

    p_terms = _probability_terms(F, G)
    s_terms = _selection_terms(A, B)
    Bp = [_apply_B(p_terms, lag) for lag in lags]
    Bs = [_apply_B(s_terms, lag) for lag in lags]
    if RR is None:
        rr = np.asarray([_rr_for_lag(A, B, lag) for lag in lags])
    else:
        if len(RR) != len(lags):
            raise ValueError("RR must match lags array in length")
        rr = RR

    count = len(lags)
    covariance_dd_dd = np.empty((count, count), dtype=np.float64)
    covariance_dd_dr = np.empty((count, count), dtype=np.float64)
    covariance_dr_dd = np.empty((count, count), dtype=np.float64)
    covariance_dr_dr = np.empty((count, count), dtype=np.float64)
    covariance_ls_numerator = np.empty((count, count), dtype=np.float64)

    if factorial_dd_normalization:
        alpha = float(N) / float(N - 1)
    else:
        alpha = 1.0

    for i, h in enumerate(lags):
        for j, k in enumerate(lags):
            trace = _trace_term(h, k, F, G, N)

            dd_dd = (
                2.0 * trace
                + 4.0 * _multinomial_bilinear(Bp[i], Bp[j], F, G, N)
            )
            dd_dr = 2.0 * _multinomial_bilinear(
                Bp[i], Bs[j], F, G, N
            )
            dr_dd = 2.0 * _multinomial_bilinear(
                Bs[i], Bp[j], F, G, N
            )
            dr_dr = _multinomial_bilinear(
                Bs[i], Bs[j], F, G, N
            )

            covariance_dd_dd[i, j] = alpha * alpha * dd_dd
            covariance_dd_dr[i, j] = alpha * dd_dr
            covariance_dr_dd[i, j] = alpha * dr_dd
            covariance_dr_dr[i, j] = dr_dr
            covariance_ls_numerator[i, j] = (
                covariance_dd_dd[i, j]
                - 2.0 * covariance_dd_dr[i, j]
                - 2.0 * covariance_dr_dd[i, j]
                + 4.0 * covariance_dr_dr[i, j]
            )

    denominator = np.outer(rr, rr)
    valid = denominator > 0.0
    covariance_ls = np.full_like(covariance_ls_numerator, np.nan)
    covariance_dd_only = np.full_like(covariance_dd_dd, np.nan)

    covariance_ls[valid] = (
        covariance_ls_numerator[valid] / denominator[valid]
    )
    covariance_dd_only[valid] = (
        covariance_dd_dd[valid] / denominator[valid]
    )

    return {
        "lags": lags,
        "RR": rr,
        "cov_DD_DD": covariance_dd_dd,
        "cov_DD_DR": covariance_dd_dr,
        "cov_DR_DD": covariance_dr_dd,
        "cov_DR_DR": covariance_dr_dr,
        "cov_LS_numerator": covariance_ls_numerator,
        "cov_LS": covariance_ls,
        "cov_DD_only": covariance_dd_only,
        "variance_LS": np.diag(covariance_ls),
        "variance_DD_only": np.diag(covariance_dd_only),
    }


# -----------------------------------------------------------------------------
# Point gridding
# -----------------------------------------------------------------------------


def grid_points(x, y, x_edges, y_edges):
    """Bin point coordinates into a two-dimensional count image.

    Parameters
    ----------
    x, y : array_like
        One-dimensional point coordinates. The arrays must have equal length.
    x_edges, y_edges : array_like
        Monotonically increasing pixel-edge coordinates.

    Returns
    -------
    numpy.ndarray
        Integer count image with shape ``(len(y_edges)-1, len(x_edges)-1)``.
        The first array axis corresponds to y and the second to x.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be equal-length one-dimensional arrays")

    counts = np.histogram2d(
        y,
        x,
        bins=(np.asarray(y_edges), np.asarray(x_edges)),
    )[0].astype(np.int64)

    if int(np.sum(counts)) != x.size:
        raise ValueError("at least one point lies outside the supplied grid")

    return counts


# -----------------------------------------------------------------------------
# Piecewise-constant selection and padded separable model
# -----------------------------------------------------------------------------


def piecewise_constant_factor(edges, intervals):
    """Create a normalized one-dimensional piecewise-constant mass.

    Parameters
    ----------
    edges : array_like
        Strictly increasing pixel-edge coordinates.
    intervals : iterable
        Sequence of ``(lower, upper, weight)`` triples. The weight is a
        nonnegative density value over the interval. Overlapping intervals add.

    Returns
    -------
    numpy.ndarray
        Normalized probability mass evaluated on the pixels defined by
        ``edges``.
    """
    edges = np.asarray(edges, dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array of length >= 2")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be strictly increasing")

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density = np.zeros_like(centers)

    for lower, upper, weight in intervals:
        if upper <= lower:
            raise ValueError("each interval must have upper > lower")
        if weight < 0.0:
            raise ValueError("interval weights must be nonnegative")

        inside = (centers >= lower) & (centers < upper)
        density[inside] += weight

    mass = density * widths
    total = float(np.sum(mass, dtype=np.float64))

    if total <= 0.0:
        raise ValueError("the piecewise-constant factor has zero mass")

    return mass / total


def separable_selection_mass(x_edges, y_edges, x_intervals, y_intervals):
    """Construct a separable selection mass from rectangular regions.

    Parameters
    ----------
    x_edges, y_edges : array_like
        Pixel-edge coordinates along x and y.
    x_intervals, y_intervals : iterable
        Piecewise-constant interval triples accepted by
        ``piecewise_constant_factor``.

    Returns
    -------
    A : numpy.ndarray
        Normalized x-axis selection factor.
    B : numpy.ndarray
        Normalized y-axis selection factor.
    selection_mass : numpy.ndarray
        Two-dimensional selection mass ``outer(B, A)``.
    """
    A = piecewise_constant_factor(x_edges, x_intervals)
    B = piecewise_constant_factor(y_edges, y_intervals)

    return A, B, np.outer(B, A)



__all__ = [
    "circular_correlation_fft_1d",
    "circular_correlation_fft_2d",
    "circular_lag_axis",
    "covariance_selected_lags",
    "empirical_pair_maps",
    "extract_lag_window",
    "grid_points",
    "model_pair_maps",
    "pad_all_arrays",
    "pad_factor",
    "pad_spatial_array",
    "padding_shape_from_lags",
    "piecewise_constant_factor",
    "required_padding_shape",
    "separable_selection_mass",
    "variance_maps",
]

