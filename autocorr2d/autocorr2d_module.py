import numpy as np
from .gridding_functions import fdft_method_default
from .utilities2d import convgrid2d, convft2d
import scipy.fft
import sys

def autocorr2d(x, y, ngx=125, ngy=250, xgmin=-200.0, xgmax=200.0, ygmax=200.0,
               method="kaiser", deconvolve=True, binconvolve="binfunc1", ninterlace=1,
               return_window=False, xwin=None, ywin=None, normwindow=False,
               patch=True, verbose=False, idltest=False):
    """Calculate 2-point autocorrelation for points at positions x,y.
    
    In typical usage for the 2-photon autocorrelation,
    x = ln(wavelength) * 3.d5 and y = time (sec)
    The ln(wavelength) scaling produces a grid which is approximately in
    1 km/s intervals.  These choices allow the xgmin, xgmax values to be
    specified in km/s and the ygmax value to be specified in seconds.
    
    This samples the unevenly spaced data to an evenly
    spaced grid so that the FFT algorithm can be used.  This
    is much faster than the direct calculation of the DFT.
    However, it does have the disadvantage that it introduces
    some aliasing.  Proper choice of the convolving function
    used in the gridding keeps the aliasing under control.
    
    This version optionally uses the interlacing approach as in
    Sefusatti et al. (2016)
    <https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.3624S/abstract>
    which doubles the number of calculations but suppresses the effects
    of aliasing.  The improvement from this approach depends on the other
    method parameters.  Currently ninterlace=4 is never significantly better
    than ninterlace=2.
    
    The default parameters give good results:
      method = "kaiser"
      binconvolVe = "binfunc1"
      deconvolve = True
      patch = True
      ninterlace = 1
    
    Derived from fdft function that performs a FFT for unevenly
    spaced data points.
    
    References:
    Schwab, F. R. 1978, VLA Scientific Memorandum No. 129, "Suppression
        of Aliasing by Convolutional Gridding Schemes".
    Greisen, E. W. 1979, VLA Scientific Memorandum No. 131, "The Effects
        of Various Convolving Functions on Aliasing and Relative 
        Signal-to-Noise Ratios".
    Schwab, F. R. 1980, VLA Scientific Memorandum No. 132, "Optimal 
        Gridding".
    Sefusatti, E., et al. 2016, MNRAS 460, 3624, "Accurate estimators of
        correlation functions in Fourier space".
    
    Required inputs:
      x       [n] x-values of photon counts
      y       [n] y-values of photon counts
              Normally y will be the times of the counts and x will be
              the 3.d5*log(wavelength) values (so that differences
              in x correspond to velocities in units of km/s)
    Optional keyword inputs:
      -------------- Grid definition: --------------
      ngx     Number of points in x grid window (default 125)
      ngy     Number of points in y grid window (default 250)
              Actual image size is [ngy+1, ngx] to account for row zero centered at 0
      xgmin, xgmax    Range for output autocorr grid (default -200, 200 km/s)
      ygmax   Range for output autocorr grid (default 200.0 sec)
              Note we always have YGMIN = 0.0
      -------------- Method controls: -------------- 
      method      Gridding interpolation function.  This must be a string or a dict with
                  a field "name" that identifies the method along with whatever additional
                  parameters are desired. For example, use
                      method=dict(name="kaiser", beta=4.09*np.pi, support=8)
                  to specify parameters for the Kaiser function.  See the fdft_method_default
                  function for the defaults.
      deconvolve  If True, deconvolve the gridding function before computing the power spectrum.
                  This is the default.  If it is explicitly set to False, the gridding function
                  remains in the power spectrum as smoothing.
      binconvolve Name of function to convolve with the power spectrum.  Default is "binfunc1".
                  To get no binning function, set binconvolve=False (or another false value).
                  Note some combinations of method/deconvolve/binconvolve are mathematically
                  equivalent (although the roundoff errors may be different).
      ninterlace  Number of interlace points.  Must be 1, 2, or 4.  Default is 1 (no interlacing).
      patch       If True, the x=0 and y=0 lines in the output get convolved with a [1,2,1] Hanning
                  filter.  This leads to much-improved quality along those lines.  Note that this has
                  no effect at all on the asymmetrical part of the autocorrelation.  The default value
                  is patch = True; set patch=False to turn this off.
      return_window  If True, also returns window function
      xwin        [2] or [:,2] array giving boundaries of regions included in x
      ywin        [2] or [:,2] array giving boundaries of regions included in y
                  If return_window is True, xwin and ywin must be specified, and a window function is
                  also computed.
      normwindow  If True, window function is normalized to match the data.  Default is window function
                  normalized to unity peak.  Note normwindow=True is exactly equivalent to multiplying the
                  window function by the square of the number of counts.
      idltest     If True, uses the IDL fftsize() function to determine good FFT sizes.  This is helpful
                  for testing comparisons with IDL results, but slows things down a little.
    Outputs:
      Function result is a [ngy+1, ngx] double array with autocorrelation in bins
      If return_window is True, also returns the matching window function derived from xwin, ywin
    Restrictions:
      x and y must be matching 1-D arrays.
    Procedure:
      Whether or not window is to be computed, xwin and ywin are used to determine the
      size of the time/wavelength windows for the calculation.  Without them, the defaults
      are to use min(x), max(x), min(y) and max(y) for the range.  Note that can lead to
      unpredictable sizes for selection, particularly for simulated data, so specifying
      xwin and ywin is recommended.
    Modification history:
      Created from IDL version, 2026 April 1
    """

    # interlace patterns
    intermap = { 1: ([0], [0]),
                 2: ([0,1], [0,1]),
                 4: ([0,1,0,1], [0,0,1,1]) }
    dxoffset, dyoffset = intermap.get(ninterlace,(None,None))
    if dxoffset is None:
        raise ValueError("ninterlace must be 1, 2 or 4")

    # get array sizes and axis limits
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1:
        raise ValueError("x must be 1-D array")
    n = x.size
    if y.ndim != 1 or y.size != n:
        raise ValueError("y must be 1-D array and must match x in length")

    if return_window:
        if xwin is None or ywin is None:
            raise ValueError("Must specify xwin, ywin parameters to compute a window function")
    if xwin is not None:
        xwin = np.asarray(xwin)
        if xwin.ndim < 1 or xwin.ndim > 2:
             raise ValueError("xwin must be [2] or [:,2] array")
        if xwin.shape[-1] != 2:
            raise ValueError("xwin must be [...,2] array")
        # change to 2-D array for consistency
        if xwin.ndim == 1:
            xwin = xwin.reshape(1,2)
    if ywin is not None:
        ywin = np.asarray(ywin)
        if ywin.ndim < 1 or ywin.ndim > 2:
            raise ValueError("ywin must be [2] or [:,2] array")
        if ywin.shape[-1] != 2:
            raise ValueError("ywin must be [...,2] array")
        if ywin.ndim == 1:
            ywin = ywin.reshape(1,2)

    # allow specifying only xgmax to get symmetrical grid about zero
    # for consistency allow only xgmin too
    if xgmin is None and xgmax is None:
        xgmin = -200.0
        xgmax = 200.0
    elif xgmin is None:
        xgmin = -xgmax
    elif xgmax is None:
        xgmax = -xgmin
    if xgmin > xgmax:
        temp = xgmin
        xgmin = xgmax
        xgmax = temp

    if ygmax is None:
        ygmax = 200.0
    ygmin = 0.0
    if ygmax <= ygmin:
        raise ValueError("Bad value for ygmax, must be > 0")

    # get the default method parameters for 2D transforms
    dmethod = fdft_method_default(method, dim=2)

    # x and y intervals in evenly sampled grid
    dx = (xgmax - xgmin) / ngx
    dy = (ygmax - ygmin) / ngy

    # data ranges
    if xwin is not None:
        xmin = xwin.min()
        xmax = xwin.max()
    else:
        xmin = x.min()
        xmax = x.max()
    if ywin is not None:
        ymin = ywin.min()
        ymax = ywin.max()
    else:
        ymin = y.min()
        ymax = y.max()
    if verbose:
        print(f"{xmin=} {xmax=} {ymin=} {ymax=} {ngx=} {ngy=} {dx=} {dy=}")

    # size of full array for gridding data
    # add enough padding to ensure there is no aliasing within the grid output region
    # XXX also add some padding for the convolution function support
    # XXX think more carefully about this, is this enough padding?
    nx = np.floor((xmax - xmin + (xgmax-xgmin))/dx).astype(int) + 1
    ny = np.floor((ymax - ymin + (ygmax-ygmin))/dy).astype(int) + 1

    # increase the grid size to make FFT efficient
    if idltest:
        # FOR TESTING: Use idl version
        nfx = fftsize(nx)
        nfy = fftsize(ny)
    else:
        nfx = scipy.fft.next_fast_len(nx, real=True)
        nfy = scipy.fft.next_fast_len(ny, real=True)

    if verbose:
        print(f"Grid size {nx} {ny} FFT size {nfx} {nfy}"
              f" array size {(nfx*nfy*8)/1.0e6:.1f} MB")

    # FFT of gridding function
    ftnorm = convft2d(nfx, nfy, method=dmethod, rfft=True)

    # do this calculation at multiple different interlace positions and average the results

    for interlace in range(ninterlace):

        xoffset = dxoffset[interlace]*dx/2
        yoffset = dxoffset[interlace]*dy/2
        if verbose:
            print(f"Interlace step {interlace+1} {xoffset=} {yoffset=}")

        # do FT of data first

        # convolve data onto grid

        cdata = convgrid2d(x + xoffset, y + yoffset, xmin, ymin, dx, dy, nfx, nfy,
                           method=dmethod, verbose=verbose)

        if verbose:
            print(f"cdata sum is {cdata.sum():.0f} {n=}")

        # transform gridded data and put results in fdata

        fdata = scipy.fft.rfft2(cdata,norm="backward")
        if verbose:
            print("Zero freq point", fdata[0,0])
        assert (ftnorm.shape==fdata.shape), f"Not right yet {cdata.shape=} {fdata.shape=} {ftnorm.shape=}"

        # divide by FT of convolving function to recover DFT of data
        # convft2d is a function which returns the FT of the convolving function
        # default is deconvolve=True, set explicitly to zero to skip this step

        if deconvolve:
            fdata /= ftnorm
            if verbose:
                print("Deconv zero freq point", fdata[0,0])

        # autocorrelation is the inverse Fourier transform of the square amplitude

        facorr = ampsq(fdata)
        if verbose:
            print("facorr zero freq point", facorr[0,0])

        # subtract self-correlations
        if deconvolve:
            facorr -= n
        else:
            facorr -= n*ftnorm^2
        if verbose:
            print("facorr zero freq after self-corr subtraction", facorr[0,0])

        if interlace == 0:
            fresult = facorr
        else:
            fresult += facorr
        if verbose:
            print(f"Completed interlace step {interlace+1}")

    fresult /= ninterlace
    facorr = fresult
    if verbose:
        print("fresult zero freq mean ", facorr[0,0])

    # if binconvolve is set, apply binning FFT
    if binconvolve:
        if binconvolve is True:
            # equivalent to binfunc1
            binfunc = "binfunc1"
        else:
            # binconvolve names the function
            binfunc = binconvolve
        facorr *= convft2d(nfx, nfy, method=binfunc, verbose=verbose, rfft=True)
        if verbose:
            print("After binconvolve facorr zero freq point", facorr[0,0])

    # specify output size in case the size is odd
    acorr = scipy.fft.irfft2(facorr,norm="backward", s=(nfy,nfx))
    if verbose:
        print("autocorr near zero freq", acorr[:2,0])

    # extract the relevant section
    ngx2 = ngx//2 # note ngx is odd, so ngx = 2*ngx2+1
    if patch:
        # extract a region with 1-pixel pads on edge, filter the x=0 and y=0 lines,
        # and then cut back to the desired region
        #
        # this is a bit messy because I want to avoid the large
        # shift, which uses a lot of memory
        # the short version would be:
        #   section1 = (shift(acorr, 1, (ngx//2)+1)) [:ngy+2, :ngx+1]
        # ... followed by the filtering and subsection extraction
        #
        section1 = np.zeros((ngy+3, ngx+2), dtype=float)
        section1[1:, ngx2+1:] = acorr[:ngy+2, :ngx2+2]
        section1[1:, :ngx2+1] = acorr[:ngy+2, nfx-ngx2-1:nfx]
        section1[0, ngx2+1:] = acorr[nfy-1, :ngx2+2]
        section1[0, :ngx2+1] = acorr[nfy-1, nfx-ngx2-1:nfx]
        # now filter the x=0 and y=0 columns and trim off the padding
        kernel = [0.25,0.5,0.25]
        section1[1, :] = np.convolve(section1[1, :], kernel, mode='same')
        section1[:, ngx2+1] = np.convolve(section1[:, ngx2+1], kernel, mode='same')
        section = section1[1:ngy+2, 1:ngx+1]
    else:
        # shift-by-hand
        # section = (shift(acorr[:ngy+1, :], 0, (ngx//2))) [:, :ngx]
        section = np.empty((ngy+1, ngx), dtype=float)
        section[:, ngx2:] = acorr[:ngy+1, :ngx-ngx2]
        section[:, :ngx2] = acorr[:ngy+1, -ngx2:]

    # we are done if the window is not requested

    if not return_window:
        return section

    # create array of covered region
    # create profiles in x and y including partially covered pixels

    xpix1 = ((xwin[:,0]-xmin)/dx).clip(min=0.0)
    xpix2 = ((xwin[:,1]-xmin)/dx).clip(max=nx-1)
    w = np.where(xpix2 > xpix1)[0]
    if len(w) == 0:
        raise ValueError("Data not covered by xwin window?")
    xpix1 = xpix1[w]
    xpix2 = xpix2[w]
    ni = len(xpix1)
    ixpix1 = np.floor(xpix1).astype(int)
    ixpix2 = np.floor(xpix2).astype(int)
    # partially covered edge fractions
    ex1 = 1.0 - (xpix1 - ixpix1)
    ex2 = xpix2 - ixpix2
    ex3 = xpix2 - xpix1
    xdata = np.zeros(nfx)
    for i in range(ni):
        i1 = ixpix1[i]
        i2 = ixpix2[i]
        if i2 == i1:
            # i1 == i2 special case
            xdata[i1] = ex3[i]
        else:
            if i2 > i1+1:
                xdata[i1+1:i2] = 1.0
            xdata[i1] = ex1[i]
            xdata[i2] = ex2[i]

    ypix1 = ((ywin[:,0]-ymin)/dy).clip(min=0.0)
    ypix2 = ((ywin[:,1]-ymin)/dy).clip(max=ny-1)
    w = np.where(ypix2 > ypix1)[0]
    if len(w) == 0:
        raise ValueError("Data not covered by ywin window?")
    ypix1 = ypix1[w]
    ypix2 = ypix2[w]
    nj = len(ypix1)
    jypix1 = np.floor(ypix1).astype(int)
    jypix2 = np.floor(ypix2).astype(int)
    ey1 = 1.0 - (ypix1 - jypix1)
    ey2 = ypix2 - jypix2
    ey3 = ypix2 - ypix1
    ydata = np.zeros(nfy)
    for j in range(nj):
        j1 = jypix1[j]
        j2 = jypix2[j]
        if j2 == j1:
            # j1 == j2 special case
            ydata[j1] = ey3[j]
        else:
            if j2 > j1+1:
                ydata[j1+1:j2] = 1.0
            ydata[j1] = ey1[j]
            ydata[j2] = ey2[j]

    cdata = np.outer(ydata, xdata) / ((ypix2-ypix1).sum()*(xpix2-xpix1).sum())
    if verbose:
        print(f"Window xdata sum {xdata.sum()} expected {(xpix2-xpix1).sum()}")
        print(f"Window ydata sum {ydata.sum()} expected {(ypix2-ypix1).sum()}")
        print(f"Window cdata sum {cdata.sum()} expected 1.0")
        print(f"{xwin=}")
        print(f"{ywin=}")
        print(f"{xpix1=}")
        print(f"{xpix2=}")
        print(f"{ypix1=}")
        print(f"{ypix2=}")
        print(f"{xdata.sum()=}")
        print(f"{ydata.sum()=}")
        print(f"{xdata.sum()*ydata.sum()=}")

    if normwindow:
        # normalize cdata to be the same as the real data
        cdata *= n

    fwcorr = ampsq(scipy.fft.rfft2(cdata,norm="backward"))
    if verbose:
        print("Window fwcorr zero freq point", fwcorr[0,0])

    #XXX??
    ## # subtract self-correlations
    ## fwcorr -= total(cdata)

    # if binconvolve is set, apply binning FFT
    if binconvolve:
        fwcorr *= convft2d(nfx, nfy, method=binfunc, verbose=verbose, rfft=True)
        if verbose:
            print("After binconvolve window fwcorr zero freq point", fwcorr[0,0])

    # specify output size in case the size is odd
    wcorr = scipy.fft.irfft2(fwcorr, norm="backward", s=(nfy,nfx))
    if verbose:
        print("wcorr near zero freq", wcorr[0:2,0])

    #XXX Need to apply patch here too??
    # shift-by-hand
    # window = (shift(wcorr[:ngy+1, :], 0, (ngx//2))) [:, :ngx]
    window = np.empty((ngy+1, ngx), dtype=float)
    window[:, ngx2:] = wcorr[:ngy+1, :ngx-ngx2]
    window[:, :ngx2] = wcorr[:ngy+1, -ngx2:]

    return section, window

def ampsq(cdata):
    """Return square amplitude of complex array"""
    return cdata.real**2 + cdata.imag**2

def fftsize(n, oned=False):
    """IDL version of best fftsize

    For testing it is helpful to have exactly the same size as IDL uses.
    For normal usage it is better to use the scipy.fft.next_fast_len() function.

    Comments from IDL version:
    Figure out good fft size for vector of length n using the IDL FFT time estimate.
    This assumes that you are doing a 2-D fft; set the ONED keyword for the 1-D length
    (which is always a power of 2, I think.)
    The lengths considered have only factors of 2 and 3.
    
    R. White, 2004 November 17
    """

    assert n > 0, "n must be greater than zero"
    if oned:
        ndim = 1
    else:
        ndim = 2

    # smallest power of 2 bigger than N
    nfac2 = np.round(np.log(n)/np.log(2)).astype(int)
    b2 = 2**nfac2
    if b2 < n:
        b2 = b2*2
        nfac2 = nfac2+1
    nfac3 = 0
    # try replacing powers of 2 with powers of 3 (which are also pretty fast in FFT)
    bbest = b2
    # estimate of execution time for IDL FFT
    tbest = float(b2)**ndim*(2.0*nfac2+12.0*nfac3+1)
    while (b2 % 2) == 0:
        if b2 >= n:
            time = float(b2)**ndim*(2.0*nfac2+12.0*nfac3+1)
            if time < tbest:
                tbest = time
                bbest = b2
            b2 = b2//2
            nfac2 = nfac2-1
        else:
            b2 = b2*3
            nfac3 = nfac3+1
    if b2 >= n:
        time = float(b2)**ndim*(2.0*nfac2+12.0*nfac3+1)
        if time < tbest:
            tbest = time
            bbest = b2
    return bbest


if __name__ == "__main__":
    method = "kaiser"

    # create test data using golden ratio spacing, which
    # is fairly random for a Fibonacci number of points

    n = 987 # fibonacci number
    phi = (1.0+np.sqrt(5.0))/2.0
    omega = (np.arange(n)*phi) % 1.0
    r = (((n + np.arange(n))*np.arange(n))*phi) % 1.0
    x = (r*np.sin(2*np.pi*omega) + 1) % 1.0
    y = (r*np.cos(2*np.pi*omega) + 1) % 1.0

    # convert to time/wavelength with reasaonble range
    time = x*400.0
    vel = (y-0.5)*500.0
    c = 3.e5
    wave = 1310.*(1+vel/c)
    zwave = np.log(wave)*c
    ywin = [0.0, 500.0]
    xwin = np.log(1310.*(1+np.array([-250.0,250.0])/c))*c

    bim, window = autocorr2d(zwave, time, method=method,
                             return_window=True, normwindow=True,
                             xwin=xwin, ywin=ywin)

    with np.printoptions(suppress=True, precision=6, floatmode="fixed", linewidth=250):
        print(f"autocorr2d {method=} {n} data points")
        print(f"xwin={np.exp(xwin/c)} (wavelength)")
        print(f"{ywin=}")
        print(f"{bim.shape=} {bim.min()=:.6f} {bim.max()=:.6f} {bim.mean()=:.6f}")
        print(f"{window.shape=} {window.min()=:.6f} {window.max()=:.6f} {window.mean()=:.6f}")
