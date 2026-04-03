import numpy as np
from scipy.ndimage import value_indices
from .gridding_functions import fdft_method_default

def convgrid(data,time,tmin,dt,m,wt=None, axis=None, normalize=True, method=None):
    """
    convolve data on unequally spaced grid to equally spaced grid
    convolved data is multiplied by a weighting function, wt, if wt is specified
    
    method is the interpolation method.  This must be a string or a dict with a field
    "name" that identifies the method along with whatever additional parameters are desired.
    For example, use method=dict(name="kaiser", beta=4.09*np.pi, support=8) to
    specify parameters for the Kaiser function.  See the fdft_method_default
    function for the defaults.
    """

    dmethod = fdft_method_default(method, dim=1)
    support = dmethod['support']

    n = len(time)
    twod = (data.ndim == 2)
    if twod:
        if axis is None:
            raise ValueError('axis must be specified for 2D data')
        if axis != 0 and axis != 1:
            raise ValueError('axis must be 0 or 1')
        if n != data.shape[axis]:
            raise ValueError(f'data dimension {axis} does not match length of time')
        n2 = data.shape[1-axis]
    elif data.ndim == 1:
        if n != data.shape[0]:
            raise ValueError('data length does not match length of time')
        n2 = 1
        axis = 0
    else:
        raise ValueError('data must be 1-D or 2-D')
    if wt is not None and (wt.ndim != 1 or wt.shape[0] != n):
        raise ValueError('wt must be same length as time')

    hsize = support/2.0
    ihsize = np.floor(hsize).astype(int)
    odd = support % 2

    u = (time-tmin)/dt
    cdata = np.zeros((n2,m))
    # loop limits plus round/floor switch work for odd or even support size
    for k in range(-ihsize+1-odd,ihsize+1):
        if odd:
            ibin = np.round(u+k).astype(int)
        else:
            ibin = np.floor(u+k).astype(int)
        sfn = dmethod['function']((u-ibin)/hsize, gridding=True, normalize=normalize, **dmethod)
        if wt is not None:
            sfn *= wt
        ibin = (ibin+m) % m
        # then loop over columns and use bincount with weights
        if axis == 0:
            if twod:
                sfn = sfn[None,:]*data
                for j in range(n2):
                    cdata[j,:] += np.bincount(ibin, minlength=m, weights=sfn[:,j])
            else:
                # data array is 1-D
                sfn *= data
                cdata[0,:] += np.bincount(ibin, minlength=m, weights=sfn)
        else:
            sfn = sfn[:,None]*data
            for j in range(n2):
                cdata[j,:] += np.bincount(ibin, minlength=m, weights=sfn[j])
    if not twod:
        # remove extraneous first dimension
        cdata = cdata[0]
    return cdata*(m/n)


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
