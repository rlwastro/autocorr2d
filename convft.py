import numpy as np
from fdft_method_default import fdft_method_default

def convft(nx, method='kaiser'):
    """return Fourier transform of convolving function
    
    method is the interpolation method.  This must be a string or a dict with a field
    "name" that identifies the method along with whatever additional parameters are desired.
    For example, use method=dict(name="kaiser", beta=4.09*np.pi, support=8) to
    specify parameters for the Kaiser function.  See the fdft_method_default.py
    function for the defaults.
    """

    dmethod = fdft_method_default(method, dim=1)
    xx = (np.arange(nx,dtype=int).clip(max=nx-np.arange(nx,dtype=int))) / (nx/2)
    gx = dmethod['function'](xx, gridding=False, **dmethod)
    return gx
