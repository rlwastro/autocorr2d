# Kaiser window function
# Partially based on kaiser function extracted from numpy
# See <https://en.wikipedia.org/wiki/Kaiser_window>
#
# R. White, 2025 February 27

import numpy as np

def kaiser(eta, beta=4.09*np.pi, support=8, normalize=False, gridding=True, **kw):
    """Return the Kaiser window.  See numpy docs for more details.

    Parameters
    ----------
    eta : float
        x/L where x is the position parameter in the window
        and L = support/2
        Note function is zero for abs(eta) > 1
    beta : float
        Shape parameter for window, = pi*alpha.
    support : int
        Size of region of non-zero support for the window
    normalize: bool
        If True, normalize so the integral of the window function is
        unity.  Default is to normalize to unity peak.
    gridding: bool
        If False, returns the Fourier transform of the window function
        instead of the window function.  Default is to return the
        window function.

    Returns
    -------
    out : array
        The window evaluated at eta, with the maximum value normalized to
        to one.

    """
    beta = float(beta)
    scalar = np.isscalar(eta)
    eta = np.atleast_1d(eta)
    if gridding:
        # window function
        out = np.zeros(eta.shape, dtype=float)
        w = np.where(abs(eta) <= 1)
        if normalize:
            ## norm = (np.exp(beta) - np.exp(-beta))*(support/2)/(4*beta)
            norm = np.sinh(beta)*support/beta
        else:
            norm = np.i0(beta)
        out[w] = np.i0(beta * np.sqrt(1-eta[w]**2))/norm
    else:
        # FT of window
        u2 = beta**2 - (0.5*support*np.pi*eta)**2
        u = np.sqrt(np.abs(u2))
        # note that eta=0 gets set to 1
        out = np.ones(u.shape, dtype=float)
        w = np.where(u2 > 0)
        out[w] = np.sinh(u[w]) / u[w]
        w = np.where(u2 < 0)
        out[w] = np.sin(u[w]) / u[w]
        # normalize to unit peak
        out *= beta/np.sinh(beta)
    if scalar:
        return out[0].item()
    else:
        return out


if __name__ == "__main__":
    xgrid = np.arange(6,dtype=float)/5.0
    print("kaiser")
    print("x  " +
          "".join([f"{x:13.3f}" for x in xgrid]))
    for gridding in [False,True]:
        for support in range(4,9):
            f = kaiser(xgrid, support=support, gridding=gridding)
            print(f"{int(gridding)} {support}" +
                  "".join([f"{x:13.8f}" for x in f]))
