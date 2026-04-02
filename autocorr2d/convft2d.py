import numpy as np
from .fdft_method_default import fdft_method_default

def convft2d(nx, ny, method="kaiser", verbose=False, rfft=False):
    """Return Fourier transform of 2D convolving function
    
    method is the interpolation method.  This must be a string or a dict with a field
    "name" that identifies the method along with whatever additional parameters are desired.
    For example, use method=dict(name="kaiser", beta=4.09*np.pi, support=8) to
    specify parameters for the Kaiser function.  See the fdft_method_default.py
    function for the defaults.

    Note x is the second (fast-changing) dimension and y is the first (slow) dimension.

    If rfft=True, this is designed to apply to the output of the rfft2 function, which
    has only half the data (so the shape is [ny, nx//2 + 1]).

    The shift and addalias code from the IDL version has been removed.
    """

    dmethod = fdft_method_default(method, dim=2)
    if verbose:
        print(f"convft2d: {dmethod=}")

    if rfft:
        xx = np.arange(nx//2 + 1) / (nx//2)
    else:
        xx = np.arange(nx).clip(max=(nx-np.arange(nx))) / (nx//2)
    yy = np.arange(ny).clip(max=(ny-np.arange(ny))) / (ny//2)
    gx = dmethod["function"](xx, gridding=False, **dmethod)
    gy = dmethod["function"](yy, gridding=False, **dmethod)
    return np.outer(gy, gx)
