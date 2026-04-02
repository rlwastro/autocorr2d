# Evaluate binning window function and its Fourier transform
#
# R. White, 2025 March 27

import numpy as np

def binfunc(eta, order, support=None, normalize=False, gridding=True, **kw):
    """Evaluate binning window function and its Fourier transform
    See this paper for details:
    <https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.3624S/abstract>
    This is a scheme with various orders of integration over top hats:
    order=1 Nearest Grid Point (NGP)
    order=2 Cloud-In-Cell (CIC)
    order=3 Triangular Shape Cloud (TSC)
    order=4 Piecewise Cubic Spline (PCS)

    Parameters
    ----------
    eta : float
        Variable which ranges from 0.0 at the centre of the
        convolution function to 1.0 at its edge. Also from 0.0
        at the center of the grid correction function to 1.0 at
        the edge of the map.
    order : int
        1, 2, 3 or 4 for the 4 orders
    support : int
        The support width of the function.  The default value is determined
        by the order, but a larger value can be specified for oversampling.
    normalize: bool
        If True, normalize the function so its integral is
        unity instead of having peak unity.  Default is False.
        This only applies if gridding is also True.
    gridding: bool
        Flag determines if the binning function or its FT should be generated.
        If True, generate the function appropriate for gridding.
        If False, generate its FT.
        (default=True)

    Returns
    -------
    out : array
        The window evaluated at eta, with the maximum value normalized to
        to one.
    """

    if not isinstance(order, int) or order < 1 or order > 4:
        raise ValueError("order must be integer between 1 and 4")
    if support is None:
        support = order
    elif not isinstance(support,int) or support < order:
        raise ValueError("support must integer >= order")
    scalar = np.isscalar(eta)
    eta = np.atleast_1d(eta)

    if gridding:
        # gridding image (window function)
        if normalize:
            # equations are already normalized to unity integral
            norm = 1.0
        else:
            # peak at zero
            norm = ([1.0, 1.0, 0.75, 2.0/3.0])[order-1]
        hsupport = 0.50*order
        aeta = hsupport*np.abs(eta)
        if order == 1:
            out = (aeta < 0.50).astype(float)
        elif order == 2:
            # out = (1.0 - aeta)*(aeta < 0.5) # XXX wrong version in paper
            out = (1.0 - aeta)*(aeta < 1.0) # my version
        elif order == 3:
            out = (
                    (0.75 - aeta**2)*(aeta < 0.5) +
                    0.5*(1.5 - aeta)**2*((aeta >= 0.5) & (aeta < 1.5))
                  )
        elif order == 4:
            out = (
                    (4.0 - 6.0*aeta**2 + 3.0*aeta**3)*(aeta < 1) +
                    (2.0 - aeta)**3*((aeta >= 1) & (aeta < 2))
                  ) / 6.0
        out = out/norm
    else:
        # Fourier transform of window function

        u = (np.pi*0.5)*eta

        # note that eta=0 gets set to 1
        out = np.ones(u.shape,dtype=float)

        w = np.where(u != 0)
        out[w] = (np.sin(u[w]) / u[w]) ** order

    if scalar:
        return out[0].item()
    else:
        return out

if __name__ == "__main__":
    xgrid = np.arange(6,dtype=float)/5.0
    print("binfunc")
    print("x  " +
          "".join([f"{x:13.3f}" for x in xgrid]))
    for gridding in [False,True]:
        for order in range(1,5):
            f = binfunc(xgrid, order=order, gridding=gridding)
            print(f"{int(gridding)} {order}" +
                  "".join([f"{x:13.8f}" for x in f]))
