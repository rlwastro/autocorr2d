import numpy as np
from .fdft_method_default import fdft_method_default
from .convgrid import convgrid
from .convft import convft
import scipy.fft
import sys

def fdft(data, time, m, *,
         axis=None, tmin=None, tmax=None, freqmin=1.0, iwttype=1, meansub=False, method=None,
         return_window=False, return_cdata=False, return_cwind=False, return_ftnorm=False,
         verbose=False):
    """
    Calculate complex discrete Fourier transform for real
    data which is sampled on unevenly spaced grid.  This
    version samples the unevenly spaced data to an evenly
    spaced grid so that the FFT algorithm can be used.  This
    is much faster than the direct calculation of the DFT.
    However, it does have the disadvantage that it introduces
    some aliasing.  Proper choice of the convolving function
    used in the gridding keeps the aliasing under control.
    
    This version also can weight the gridded data by the inverse
    of the number of nearby data points, which should remove
    some of the artifacts due to uneven sampling (see Schwab 
    1978).  In VLA parlance that is uniform weighting
    instead of natural weighting.
    
    References:
        Schwab, F. R. 1978, VLA Scientific Memorandum No. 129, "Suppression
            of Aliasing by Convolutional Gridding Schemes".
        Greisen, E. W. 1979, VLA Scientific Memorandum No. 131, "The Effects
            of Various Convolving Functions on Aliasing and Relative 
            Signal-to-Noise Ratios".
        Schwab, F. R. 1980, VLA Scientific Memorandum No. 132, "Optimal 
            Gridding".
    Usage:
        fdata = fdft(data,time,mpts)
        or
        fdata, fwind = fdft(data,time,mpts,return_iindow=True)
    
    Inputs:
        data    [n] Array to be transformed.  If axis is specified,
                this can also be a [n,:] or [:,n] array (for axis=0 or 1)
                of time-sampled spectra.
        time    [n] Times of data samples
        m       Number of frequency points in transform
    Optional keyword inputs:
        axis    For 2-D data, specifies which axis is the time dimension.
                This tries to guess the value for axis by looking for
                a dimension that matches the time array, but if the
                results are ambiguous then axis must be specified as
                either 0 or 1 for the 1st or 2nd dimension.
        tmin    Minimum time for resampled range (default is time[0])
        tmax    Maximum time for resampled range (default is time[n-1])
        freqmin Frequency interval in frequency grid.  Grid goes 
                from 0 to (m-1)*(tmax-tmin)*freqmin.
                Default is freqmin=1, which is the usual grid spacing
                for a FFT of equally spaced data.
        iwttype Weighting type (0=uniform, 1=natural, default is 1)
        meansub If true, subtract mean from gridded time series before taking FFT
        method  Interpolation method.  This must be a string or a dict with a field
                "name" that identifies the method along with whatever additional parameters are desired.
                For example, use method=dict(name="kaiser", beta=4.09*np.pi, support=8) to
                specify parameters for the Kaiser function.  See the fdft_method_default.py
                function for the defaults.
        verbose Set to true to print some info
    Outputs:
        Function result is a [m] (or [:,m] for 2-D data) complex array with
            Fourier transform
        If return_window is true, also returns [2*m] complex array with window function
    Restrictions:
        data and time must be matching 1-D arrays.  If data is 2-D,
        one of the dimensions (specified by axis) must match the time array.
    Modification history:
        Fortran version:
            Programmer: R. White        10 July 1985
            Modified for H. Bond        23 May  1986
        IDL version:
            R. White, 2016 March 22
            Added AXIS to support 2-D arrays, 2016 March 29
            Converted to METHOD variable to pass parameters for kernel, 2025 March 3
        Python version:
            R. White, 2026 March 23
    """

    # get array sizes and time axis
    if hasattr(time, 'ndim') and time.ndim == 1:
        n = time.shape[0]
    else:
        raise ValueError('time must be 1-D array')
    if not (hasattr(data,'ndim') and data.ndim in (0,1)):
        raise ValueError("data must be numpy 1-D or 2-D array")
    if data.ndim == 1:
        twod = False
        if data.shape[0] != n:
            raise ValueError('1-D data array must match time in length')
        if axis is None:
            axis = 0
        elif axis != 0:
            raise ValueError(f'axis={axis} is not compatible with 1-D data input')
    else:
        twod = True
        if axis is not None:
            if axis not in (0,1):
                raise ValueError('axis must be 0 or 1')
            if data.shape[axis] != n:
                raise ValueError(f'Length of data axis={axis} must match time')
        else:
            if data.shape[0] == n:
                axis = 0
                if data.shape[1] == n:
                    raise ValueError('Error: ambiguous time axis for square data array, must specify axis value')
            elif data.shape[1] == n:
                axis = 1
            else:
                raise ValueError('Error: no dimension of data matches the length of time')
        # n2 is the other dimension
        n2 = data.shape[1-axis]

    if tmin is None:
        tmin = time.min()
    if tmax is None:
        tmax = time.max()

    # get the default method parameters for 1D transforms
    dmethod = fdft_method_default(method, dim=1)

    # 0 = uniform weighting, 1 = natural weighting
    wtmap = {0: True, 1: False}
    uniform = wtmap.get(iwttype)
    if uniform is None:
        raise ValueError(f"Unknown weight type, iwttype={iwttype}")

    # time interval in evenly sampled grid

    dt = (tmax-tmin) / (freqmin*m)

    # calculate weighting function from time grid

    if uniform:
        if verbose:
            print('Using uniform weighting for gridding')
        ibin = np.floor((time-tmin)/dt).astype(int)
        h = np.bincount(ibin,minlength=m+1)
        w = np.where((ibin >= 0) & (ibin <= m))[0]
        if len(w) == n:
            wt = 1.0/h[ibin]
        else:
            print(f'*** Warning: {n-nw} points fall outside grid')
            print('*** Need to decrease frequency interval')
            wt = np.zeros(n)
            wt[w] = 1.0/h[ibin[w]]
        # divide by the mean weights to normalize everything properly 
        wt = wt/wt.mean()
    else:
        # natural weighting -- no weights needed unless window is requested
        if verbose:
            print('Using natural weighting for gridding')
        if return_window:
            wt = np.ones(n)

    # remove mean before gridding

    if meansub:
        # subtract mean from each vector
        cmean = data.mean(axis=axis)
        if twod:
            if axis == 0:
                cmean = cmean[None,:]
            else:
                cmean = cmean[:,None]

    # do FT of data first

    # convolve data onto grid using weighting function in WIND

    if uniform:
        if meansub:
            cdata = convgrid(data-cmean, time, tmin, dt, m, wt=wt, axis=axis, method=dmethod)
        else:
            cdata = convgrid(data, time, tmin, dt, m, wt=wt, axis=axis, method=dmethod)
    else:
        if meansub:
            cdata = convgrid(data-cmean, time, tmin, dt, m, axis=axis, method=dmethod)
        else:
            cdata = convgrid(data, time, tmin, dt, m, axis=axis, method=dmethod)

    # transform gridded data and put results in fdata

    # fdata = scipy.fft.rfft(cdata,norm="forward",axis=-1)
    fdata = scipy.fft.fft(cdata,norm="forward",axis=-1)

    # divide by FT of convolving function to recover DFT of data
    # convft is a function which returns the FT of the convolving function

    ftnorm = convft(m, method=dmethod)
    # XXX THIS IS WRONG because of the shape of the rfft result
    # XXX switched to the regular (not real) fft to work around that for now
    if twod:
        for j in range(n2):
            fdata[j] /= ftnorm
    else:
        fdata /= ftnorm

    # now do same for window
    # for window, data=weighting function and no extra weighting is used

    if return_window:
        cwind = convgrid(wt, time, tmin, dt/2.0, 2*m, method=method)
        # fwind = scipy.fft.rfft(cwind,norm="forward",axis=-1)
        fwind = scipy.fft.fft(cwind,norm="forward",axis=-1)
        fwind /= convft(2*m, method=method)
    rv = [fdata]
    if return_window:
        rv.append(fwind)
    if return_cdata:
        rv.append(cdata)
    if return_cwind:
        rv.append(cwind)
    if return_ftnorm:
        rv.append(ftnorm)
    if len(rv)==1:
        return rv[0]
    else:
        return tuple(rv)

if __name__ == "__main__":
    from printit import printit

    # create test data for fdft

    n = 55 # fibonacci number
    # golden ratio spacing is fairly random
    phi = (1.0+np.sqrt(5.0))/2.0
    time = (np.arange(n,dtype=float)*phi) % 1.0
    ## time = np.sort(time) # sort by time (not required)
    data = np.sin(5*2*np.pi*time)

    if len(sys.argv) <= 1:
        methods = ["kaiser"]
    else:
        methods = sys.argv[1:]

    for method in methods:
        fdata, fwind, cdata, cwind, ftnorm = fdft(data, time, 64,
                                                  tmin=0.0, tmax=1.0,
                                                  method=method,
                                                  return_window=True,
                                                  return_cdata=True,
                                                  return_cwind=True,
                                                  return_ftnorm=True)
        print("fdft", method)
        printit("data", data)
        printit("time", time)
        printit("fdata", fdata)
        printit("fwind", fwind)
        printit("cdata", cdata)
        printit("cwind", cwind)
        printit("ftnorm", ftnorm)
