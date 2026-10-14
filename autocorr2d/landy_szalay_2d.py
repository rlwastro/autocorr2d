"""Anisotropic 2-D Landy-Szalay statistics for zero-padded arrays.

The pair-map functions operate on externally supplied padded 2-D count and
selection arrays. The variance and covariance functions operate on externally
supplied padded 1-D separable factors

    d(x, y) = f(x) g(y),
    r(x, y) = a(x) b(y).

All probability arrays contain probability mass per pixel. Physical values are
normalized internally unless normalization is disabled. Only NumPy is needed.
"""

import numpy as np
import scipy.fft


def _physical_slice(padded_size, support_size, support_origin):
    """Return the physical slice after validating one-dimensional geometry."""
    padded_size = int(padded_size)
    support_size = int(support_size)
    support_origin = int(support_origin)

    if support_size <= 0:
        raise ValueError("support_size must be positive")

    if support_origin < 0 or support_origin + support_size > padded_size:
        raise ValueError("the physical support does not fit in the padded array")

    if padded_size < 2 * support_size - 1:
        raise ValueError("padded size must be at least 2 * support_size - 1")

    return slice(support_origin, support_origin + support_size)


def _physical_slices(padded_shape, support_shape, support_origin):
    """Return physical slices after validating two-dimensional geometry."""
    if len(padded_shape) != 2:
        raise ValueError("padded arrays must be two-dimensional")

    if len(support_shape) != 2 or len(support_origin) != 2:
        raise ValueError("support_shape and support_origin must have length two")

    y_slice = _physical_slice(
        padded_shape[0],
        support_shape[0],
        support_origin[0],
    )
    x_slice = _physical_slice(
        padded_shape[1],
        support_shape[1],
        support_origin[1],
    )
    return y_slice, x_slice


def _validate_zero_padding(array, physical_slice, name, tolerance):
    """Verify that a one-dimensional array is zero outside its support."""
    if tolerance is None:
        return

    outside = np.array(array, copy=True)
    outside[physical_slice] = 0
    maximum = np.max(np.abs(outside), initial=0.0)

    if maximum > tolerance:
        raise ValueError(name + " is not zero outside its physical support")


def _validate_zero_padding_2d(array, physical_slices, name, tolerance):
    """Verify that a two-dimensional array is zero outside its support."""
    if tolerance is None:
        return

    outside = np.array(array, copy=True)
    outside[physical_slices] = 0
    maximum = np.max(np.abs(outside), initial=0.0)

    if maximum > tolerance:
        raise ValueError(name + " is not zero outside its physical support")


def _prepare_padded_factor(
    factor_padded,
    support_size,
    support_origin,
    name,
    normalize,
    check_padding,
    zero_tolerance,
):
    """Validate a padded 1-D factor and return its normalized physical part."""
    factor_padded = np.asarray(factor_padded, dtype=float)

    if factor_padded.ndim != 1 or not np.isfinite(factor_padded).all():
        raise ValueError(name + " must be a finite one-dimensional array")

    if np.any(factor_padded < 0):
        raise ValueError(name + " must be nonnegative")

    physical_slice = _physical_slice(
        factor_padded.size,
        support_size,
        support_origin,
    )

    if check_padding:
        _validate_zero_padding(
            factor_padded,
            physical_slice,
            name,
            zero_tolerance,
        )

    factor = np.array(factor_padded[physical_slice], copy=True)
    total = factor.sum()

    if total <= 0:
        raise ValueError(name + " must have positive mass")

    if normalize:
        factor /= total
    elif not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
        raise ValueError(name + " must sum to one over its physical support")

    return factor


def _prepare_separable_factors(
    f_padded,
    g_padded,
    a_padded,
    b_padded,
    support_shape,
    support_origin,
    normalize_factors,
    check_padding,
    zero_tolerance,
):
    """Validate and extract the four separable one-dimensional factors."""
    ny, nx = map(int, support_shape)
    y_origin, x_origin = map(int, support_origin)

    f = _prepare_padded_factor(
        f_padded,
        nx,
        x_origin,
        "f_padded",
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
    g = _prepare_padded_factor(
        g_padded,
        ny,
        y_origin,
        "g_padded",
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
    a = _prepare_padded_factor(
        a_padded,
        nx,
        x_origin,
        "a_padded",
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
    b = _prepare_padded_factor(
        b_padded,
        ny,
        y_origin,
        "b_padded",
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
    return f, g, a, b


def lag_axes(support_shape):
    """Return full integer y- and x-lag axes.

    Parameters
    ----------
    support_shape : tuple of int
        Unpadded physical shape ``(ny, nx)``.

    Returns
    -------
    lag_y, lag_x : ndarray
        Full physical lag axes.
    """
    ny, nx = map(int, support_shape)
    return np.arange(-ny + 1, ny), np.arange(-nx + 1, nx)


def padded_linear_correlation(
    first_padded,
    second_padded,
    support_shape,
    support_origin=(0, 0),
    crop=True,
    check_padding=False,
    zero_tolerance=0.0,
):
    """Compute a linear 2-D correlation using already padded input arrays.

    Parameters
    ----------
    first_padded, second_padded : array_like
        Common-shape, already zero-padded two-dimensional arrays.
    support_shape : tuple of int
        Unpadded physical shape ``(ny, nx)``.
    support_origin : tuple of int, optional
        Upper-left index ``(y0, x0)`` of the physical support.
    crop : bool, optional
        If True, return the physical ``(2*ny-1, 2*nx-1)`` lag rectangle.
    check_padding : bool, optional
        If True, verify zeros outside the physical support.
    zero_tolerance : float or None, optional
        Absolute tolerance for the padding check. None disables the check.

    Returns
    -------
    correlation : ndarray
        FFT-shifted linear correlation. Zero lag is central when cropped.
    """
    first_padded = np.asarray(first_padded, dtype=float)
    second_padded = np.asarray(second_padded, dtype=float)

    if first_padded.shape != second_padded.shape:
        raise ValueError("the padded arrays must have identical shapes")

    if not np.isfinite(first_padded).all() or not np.isfinite(second_padded).all():
        raise ValueError("the padded arrays must contain only finite values")

    physical_slices = _physical_slices(
        first_padded.shape,
        support_shape,
        support_origin,
    )

    if check_padding:
        _validate_zero_padding_2d(
            first_padded,
            physical_slices,
            "first_padded",
            zero_tolerance,
        )
        _validate_zero_padding_2d(
            second_padded,
            physical_slices,
            "second_padded",
            zero_tolerance,
        )

    first_fft = scipy.fft.rfft2(first_padded, norm="backward")
    second_fft = scipy.fft.rfft2(second_padded, norm="backward")
    correlation = scipy.fft.fftshift(
        scipy.fft.irfft2(np.conj(first_fft) * second_fft, norm="backward",
                         s=first_padded.shape)
    )

    if not crop:
        return correlation

    ny, nx = map(int, support_shape)
    center_y, center_x = np.array(first_padded.shape) // 2
    return correlation[
        center_y - ny + 1 : center_y + ny,
        center_x - nx + 1 : center_x + nx,
    ]


def pair_maps_from_padded(
    counts_padded,
    selection_mass_padded,
    support_shape,
    poisson_mean=None,
    support_origin=(0, 0),
    normalize_selection=True,
    check_padding=False,
    zero_tolerance=0.0,
):
    """Compute DD, directed and symmetric DR, and RR from padded 2-D arrays.

    Parameters
    ----------
    counts_padded : array_like
        Already padded two-dimensional count image.
    selection_mass_padded : array_like
        Already padded two-dimensional selection/random mass image.
    support_shape : tuple of int
        Unpadded physical shape ``(ny, nx)``.
    poisson_mean : float or None, optional
        Fixed Poisson mean Lambda. If None, use observed-N normalization.
    support_origin : tuple of int, optional
        Upper-left physical-support index in both padded arrays.
    normalize_selection : bool, optional
        Normalize selection mass to sum to one over the physical support.
    check_padding : bool, optional
        Verify zeros outside the support.
    zero_tolerance : float or None, optional
        Padding-check tolerance. None disables the numerical check.

    Returns
    -------
    result : dict
        Pair maps, lag axes, observed count, and normalization count.
    """
    counts_padded = np.asarray(counts_padded, dtype=float)
    selection_mass_padded = np.asarray(selection_mass_padded, dtype=float)

    if counts_padded.shape != selection_mass_padded.shape:
        raise ValueError("count and selection padded shapes must match")

    if np.any(counts_padded < 0) or np.any(selection_mass_padded < 0):
        raise ValueError("counts and selection mass must be nonnegative")

    physical_slices = _physical_slices(
        counts_padded.shape,
        support_shape,
        support_origin,
    )

    if check_padding:
        _validate_zero_padding_2d(
            counts_padded,
            physical_slices,
            "counts_padded",
            zero_tolerance,
        )
        _validate_zero_padding_2d(
            selection_mass_padded,
            physical_slices,
            "selection_mass_padded",
            zero_tolerance,
        )

    selection_mass_padded = np.array(selection_mass_padded, copy=True)
    selection_total = selection_mass_padded[physical_slices].sum()

    if selection_total <= 0:
        raise ValueError("selection_mass_padded must have positive mass")

    if normalize_selection:
        selection_mass_padded /= selection_total
    elif not np.isclose(selection_total, 1.0, rtol=1e-12, atol=1e-15):
        raise ValueError("selection mass must sum to one")

    observed_count = counts_padded[physical_slices].sum()

    if observed_count < 2:
        raise ValueError("at least two observed points are required")

    correlation_arguments = {
        "support_shape": support_shape,
        "support_origin": support_origin,
        "crop": True,
        "check_padding": False,
    }
    raw_dd = padded_linear_correlation(
        counts_padded,
        counts_padded,
        **correlation_arguments,
    )
    raw_dd[tuple(np.array(raw_dd.shape) // 2)] -= observed_count

    if poisson_mean is None:
        normalization_count = observed_count
        dd_denominator = observed_count * (observed_count - 1)
        data_denominator = observed_count
    else:
        normalization_count = float(poisson_mean)

        if not np.isfinite(normalization_count) or normalization_count <= 0:
            raise ValueError("poisson_mean must be finite and positive")

        dd_denominator = normalization_count**2
        data_denominator = normalization_count

    dd = raw_dd / dd_denominator
    data_mass_padded = counts_padded / data_denominator
    dr_directed = padded_linear_correlation(
        data_mass_padded,
        selection_mass_padded,
        **correlation_arguments,
    )
    rr = padded_linear_correlation(
        selection_mass_padded,
        selection_mass_padded,
        **correlation_arguments,
    )
    dr_symmetric = 0.5 * (dr_directed + dr_directed[::-1, ::-1])
    lag_y, lag_x = lag_axes(support_shape)

    return {
        "dd": dd,
        "dr_directed": dr_directed,
        "dr_symmetric": dr_symmetric,
        "rr": rr,
        "lag_y": lag_y,
        "lag_x": lag_x,
        "observed_count": observed_count,
        "normalization_count": normalization_count,
    }


def _lag_rectangle(
    support_shape,
    x_lag_min,
    x_lag_max,
    y_lag_min,
    y_lag_max,
):
    """Create an inclusive rectangular lag list and its coordinate axes."""
    ny, nx = map(int, support_shape)
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
    return lags, lag_y, lag_x


def _shift_1d(array, lag):
    """Return output[i] = array[i + lag], with zero extension."""
    output = np.zeros_like(array)
    size = array.size
    input_start = max(0, lag)
    input_stop = min(size, size + lag)
    output_start = max(0, -lag)
    output_stop = min(size, size - lag)

    if input_stop > input_start:
        output[output_start:output_stop] = array[input_start:input_stop]

    return output


def _direct_autocorrelation_values(array, lags):
    """Compute selected one-dimensional autocorrelation values directly."""
    values = np.empty(len(lags), dtype=float)
    size = array.size

    for index, lag in enumerate(lags):
        if lag >= 0:
            values[index] = np.dot(array[: size - lag], array[lag:])
        else:
            values[index] = np.dot(array[-lag:], array[: size + lag])

    return values


def _fft_autocorrelation_values(
    physical_array,
    padded_size,
    support_origin,
    lags,
):
    """Compute selected one-dimensional autocorrelation values by FFT."""
    padded = np.zeros(int(padded_size), dtype=float)
    physical_slice = slice(
        int(support_origin),
        int(support_origin) + physical_array.size,
    )
    padded[physical_slice] = physical_array
    transform = scipy.fft.rfft(padded, norm="backward")
    correlation = scipy.fft.fftshift(
        scipy.fft.irfft(np.conj(transform) * transform, norm="backward",
                        n=padded.shape[0])
    )
    center = padded.size // 2
    return np.array([correlation[center + lag] for lag in lags])


def _choose_1d_method(method, physical_size, padded_size, lags):
    """Choose direct sums or FFT for one-dimensional autocorrelations."""
    if method not in ("auto", "direct", "fft"):
        raise ValueError("method must be 'auto', 'direct', or 'fft'")

    if method != "auto":
        return method

    direct_work = np.sum(physical_size - np.abs(lags))
    fft_work = 2.0 * padded_size * np.log2(max(padded_size, 2))

    if direct_work <= fft_work:
        return "direct"

    return "fft"


def _autocorrelation_values(
    array,
    padded_size,
    support_origin,
    lags,
    method,
):
    """Compute selected 1-D autocorrelations with the requested method."""
    selected_method = _choose_1d_method(
        method,
        array.size,
        padded_size,
        lags,
    )

    if selected_method == "direct":
        values = _direct_autocorrelation_values(array, lags)
    else:
        values = _fft_autocorrelation_values(
            array,
            padded_size,
            support_origin,
            lags,
        )

    return values, selected_method


def _lag_components(f, g, a, b, dy, dx, estimator):
    """Return separable signed components for one influence function."""
    if dy == 0 and dx == 0:
        x_parts = [_shift_1d(f, 0)]
        y_parts = [_shift_1d(g, 0)]
        coefficients = [1.0]

        if estimator == "landy_szalay":
            x_parts.append(_shift_1d(a, 0))
            y_parts.append(_shift_1d(b, 0))
            coefficients.append(-1.0)
    else:
        x_parts = [
            _shift_1d(f, dx),
            _shift_1d(f, -dx),
        ]
        y_parts = [
            _shift_1d(g, dy),
            _shift_1d(g, -dy),
        ]
        coefficients = [1.0, 1.0]

        if estimator == "landy_szalay":
            x_parts.extend(
                [
                    _shift_1d(a, dx),
                    _shift_1d(a, -dx),
                ]
            )
            y_parts.extend(
                [
                    _shift_1d(b, dy),
                    _shift_1d(b, -dy),
                ]
            )
            coefficients.extend([-1.0, -1.0])

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


def _separable_q_diagonal(f, g, a, b, lags, estimator):
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
            estimator,
        )
        values[index] = _separable_q_entry(
            f,
            g,
            components,
            components,
        )

    return values


def _same_symmetric_lag(first_lag, second_lag):
    """Return True if two lags define the same symmetrized displacement."""
    first_dy, first_dx = first_lag
    second_dy, second_dx = second_lag
    return (
        first_dy == second_dy
        and first_dx == second_dx
    ) or (
        first_dy == -second_dy
        and first_dx == -second_dx
    )


def variances_from_padded(
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
    support_origin=(0, 0),
    normalize_factors=True,
    method="auto",
    rr_min_fraction=1e-10,
    check_padding=False,
    zero_tolerance=0.0,
):
    """Compute LS and DD-only variance maps from padded separable factors.

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
    support_origin : tuple of int, optional
        Physical origin ``(y_origin, x_origin)`` in the padded factors.
    normalize_factors : bool, optional
        Normalize each physical one-dimensional factor to sum to one.
    method : {'auto', 'direct', 'fft'}, optional
        Method for the one-dimensional autocorrelations needed by DD and RR.
        With ``auto``, x and y methods are selected independently.
    rr_min_fraction : float, optional
        Mask RR values below this fraction of maximum requested RR.
    check_padding : bool, optional
        Verify zeros outside each physical support.
    zero_tolerance : float or None, optional
        Padding-check tolerance. None disables the numerical check.

    Returns
    -------
    result : dict
        Variance maps and their Q, shot-noise, RR, lag, and method components.

    Notes
    -----
    The Q terms are computed with one-dimensional separable products. The DD
    population and RR terms factor into x and y autocorrelations. Consequently,
    no two-dimensional variance array is shifted or transformed.
    """
    f, g, a, b = _prepare_separable_factors(
        f_padded,
        g_padded,
        a_padded,
        b_padded,
        support_shape,
        support_origin,
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
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
    x_origin = support_origin[1]
    y_origin = support_origin[0]

    ff, method_x = _autocorrelation_values(
        f,
        np.asarray(f_padded).size,
        x_origin,
        unique_x_lags,
        method,
    )
    aa, method_x_selection = _autocorrelation_values(
        a,
        np.asarray(a_padded).size,
        x_origin,
        unique_x_lags,
        method,
    )
    gg, method_y = _autocorrelation_values(
        g,
        np.asarray(g_padded).size,
        y_origin,
        unique_y_lags,
        method,
    )
    bb, method_y_selection = _autocorrelation_values(
        b,
        np.asarray(b_padded).size,
        y_origin,
        unique_y_lags,
        method,
    )

    if method_x != method_x_selection or method_y != method_y_selection:
        raise RuntimeError("inconsistent automatic method selection")

    dd_population = np.outer(gg, ff)
    rr = np.outer(bb, aa)
    q_dd = _separable_q_diagonal(
        f,
        g,
        a,
        b,
        lags,
        "dd",
    ).reshape(rr.shape)
    q_landy_szalay = _separable_q_diagonal(
        f,
        g,
        a,
        b,
        lags,
        "landy_szalay",
    ).reshape(rr.shape)
    p_shot_noise = 0.5 * dd_population
    zero_y = np.where(lag_y == 0)[0]
    zero_x = np.where(lag_x == 0)[0]

    if zero_y.size and zero_x.size:
        p_shot_noise[zero_y[0], zero_x[0]] = dd_population[
            zero_y[0],
            zero_x[0],
        ]

    variance_landy_szalay = (
        q_landy_szalay / poisson_mean
        + 2 * p_shot_noise / poisson_mean**2
    ) / rr**2
    variance_dd_only = (
        q_dd / poisson_mean
        + 2 * p_shot_noise / poisson_mean**2
    ) / rr**2
    valid = rr > rr.max() * rr_min_fraction
    variance_landy_szalay[~valid] = np.nan
    variance_dd_only[~valid] = np.nan

    return {
        "variance_landy_szalay": variance_landy_szalay,
        "variance_dd_only": variance_dd_only,
        "rr": rr,
        "dd_population": dd_population,
        "q_landy_szalay": q_landy_szalay,
        "q_dd": q_dd,
        "p_shot_noise": p_shot_noise,
        "valid": valid,
        "lag_y": lag_y,
        "lag_x": lag_x,
        "lags": lags,
        "method_x": method_x,
        "method_y": method_y,
    }


def complete_covariances_from_padded(
    f_padded,
    g_padded,
    a_padded,
    b_padded,
    support_shape,
    poisson_mean,
    lags,
    support_origin=(0, 0),
    normalize_factors=True,
    method="auto",
    check_padding=True,
    zero_tolerance=0.0,
):
    """Compute complete LS and DD-only covariance matrices separably.

    Parameters
    ----------
    f_padded : array_like
        Padded x factor of the data probability mass.
    g_padded : array_like
        Padded y factor of the data probability mass.
    a_padded : array_like
        Padded x factor of the selection probability mass.
    b_padded : array_like
        Padded y factor of the selection probability mass.
    support_shape : tuple of int
        Unpadded two-dimensional shape ``(ny, nx)``.
    poisson_mean : float
        Fixed expected Poisson count Lambda.
    lags : sequence of (int, int)
        Exact symmetrized pixel lags to include.
    support_origin : tuple of int, optional
        Physical origin ``(y_origin, x_origin)`` in the padded factors.
    normalize_factors : bool, optional
        Normalize each physical factor to sum to one.
    method : {'auto', 'direct', 'fft'}, optional
        Method for one-dimensional autocorrelations used by DD and RR.
    check_padding : bool, optional
        Verify zeros outside each physical support.
    zero_tolerance : float or None, optional
        Padding-check tolerance. None disables the numerical check.

    Returns
    -------
    result : dict
        Complete LS and DD-only covariance matrices and all component matrices.

    Notes
    -----
    Every Q entry is a sum of products of one-dimensional x and y integrals.
    This avoids constructing K two-dimensional influence images. The output is
    still a dense K by K covariance matrix.
    """
    f, g, a, b = _prepare_separable_factors(
        f_padded,
        g_padded,
        a_padded,
        b_padded,
        support_shape,
        support_origin,
        normalize_factors,
        check_padding,
        zero_tolerance,
    )
    poisson_mean = float(poisson_mean)

    if not np.isfinite(poisson_mean) or poisson_mean <= 0:
        raise ValueError("poisson_mean must be finite and positive")

    lags = np.asarray(lags, dtype=int)

    if lags.ndim != 2 or lags.shape[1] != 2:
        raise ValueError("lags must be a sequence of (dy, dx) pairs")

    ny, nx = map(int, support_shape)

    if np.any(np.abs(lags[:, 0]) >= ny) or np.any(np.abs(lags[:, 1]) >= nx):
        raise ValueError("a requested lag is outside the physical range")

    unique_x_lags, x_inverse = np.unique(lags[:, 1], return_inverse=True)
    unique_y_lags, y_inverse = np.unique(lags[:, 0], return_inverse=True)
    ff, method_x = _autocorrelation_values(
        f,
        np.asarray(f_padded).size,
        support_origin[1],
        unique_x_lags,
        method,
    )
    aa, method_x_selection = _autocorrelation_values(
        a,
        np.asarray(a_padded).size,
        support_origin[1],
        unique_x_lags,
        method,
    )
    gg, method_y = _autocorrelation_values(
        g,
        np.asarray(g_padded).size,
        support_origin[0],
        unique_y_lags,
        method,
    )
    bb, method_y_selection = _autocorrelation_values(
        b,
        np.asarray(b_padded).size,
        support_origin[0],
        unique_y_lags,
        method,
    )

    if method_x != method_x_selection or method_y != method_y_selection:
        raise RuntimeError("inconsistent automatic method selection")

    dd_values = ff[x_inverse] * gg[y_inverse]
    rr_values = aa[x_inverse] * bb[y_inverse]
    dd_components = [
        _lag_components(f, g, a, b, dy, dx, "dd")
        for dy, dx in lags
    ]
    ls_components = [
        _lag_components(f, g, a, b, dy, dx, "landy_szalay")
        for dy, dx in lags
    ]
    number_of_lags = len(lags)
    q_dd = np.empty((number_of_lags, number_of_lags), dtype=float)
    q_landy_szalay = np.empty_like(q_dd)

    for row in range(number_of_lags):
        for column in range(row, number_of_lags):
            q_dd_value = _separable_q_entry(
                f,
                g,
                dd_components[row],
                dd_components[column],
            )
            q_ls_value = _separable_q_entry(
                f,
                g,
                ls_components[row],
                ls_components[column],
            )
            q_dd[row, column] = q_dd_value
            q_dd[column, row] = q_dd_value
            q_landy_szalay[row, column] = q_ls_value
            q_landy_szalay[column, row] = q_ls_value

    p_shot_noise = np.zeros_like(q_dd)

    for row in range(number_of_lags):
        for column in range(number_of_lags):
            if _same_symmetric_lag(lags[row], lags[column]):
                if lags[row, 0] == 0 and lags[row, 1] == 0:
                    factor = 1.0
                else:
                    factor = 0.5

                p_shot_noise[row, column] = factor * dd_values[row]

    cov_dd_pairs = (
        q_dd / poisson_mean
        + 2 * p_shot_noise / poisson_mean**2
    )
    cov_landy_szalay_numerator = (
        q_landy_szalay / poisson_mean
        + 2 * p_shot_noise / poisson_mean**2
    )
    rr_outer = np.outer(rr_values, rr_values)

    return {
        "lags": lags,
        "rr": rr_values,
        "dd_population": dd_values,
        "q_dd": q_dd,
        "q_landy_szalay": q_landy_szalay,
        "p_shot_noise": p_shot_noise,
        "cov_dd_pairs": cov_dd_pairs,
        "cov_dd_only": cov_dd_pairs / rr_outer,
        "cov_landy_szalay": cov_landy_szalay_numerator / rr_outer,
        "method_x": method_x,
        "method_y": method_y,
    }


__all__ = [
    "lag_axes",
    "padded_linear_correlation",
    "pair_maps_from_padded",
    "variances_from_padded",
    "complete_covariances_from_padded",
]
