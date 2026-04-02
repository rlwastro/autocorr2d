# autocorr2d: 2-photon autocorrelation function

This repository includes code for creating the 2-photon autocorrelation function from time-tagged data.  It also includes code for calculating fast Fourier transforms of unevenly sampled data.

## Installation

Installable using pip, uv or other tools.  For example:

```
uv add https://github.com/rlwastro/autocorr2d.git
```
will install the package and its dependencies if necessary (only numpy and scipy are required, and it probably will work with older versions of those packages but has not been tested on them).

## Usage: fdft, fast discrete Fourier transform for unevenly spaced data

The `fdft()` function calculates the complex discrete Fourier transform for real
data that is sampled on an unequally spaced grid.  The standard FFT algorithm requires
equally spaced sample.  A direct sum of the Fourier series can be used for unequally
spaced data points (where each point comes with a time and a data value).  But such a
discrete Fourier transform (DFT) is extremely slow to compute.

The radio astronomers at the Very Large Array solved this problem for interferometric
data 50 years ago. Schwab et al. (see references below) showed that the randomly placed data
points can be sampled onto a regular grid using a convolving kernel, and then a standard FFT can be used
on the evenly spaced samples.  Then the FFT output can be corrected for the convolving kernel by
dividing the FFT by the Fourier transform of the kernel.  The result is very fast to compute and
is a very accurate representation of the exact DFT result.

The Schwab papers determined the best choice of convolving kernel that is both compact (zero outside
some fairly small region, making the convolution onto the grid fast) and that does a good job of
suppressing aliasing (which is the source of the small differences compared with the DFT). 

The `fdft()` function implements this algorithm for a 1-dimensional time series. The usage is simple:

```python
from autocorr2d import dft

fdata = fdft(data,time,mpts)

# or if you also want the window function
fdata, fwind = fdft(data,time,mpts,return_window=True)
```
This returns a FT with `mpts` points.  By default the sampling time range is determined by
the min and max of the time array, but there are optional parameters to change the time limits
and frequency spacing.

There is a choice of convolving functions, specified by the `method` parameter.  The prolate spheroidal function of
order 0 (identified by Schwab as ideal) is available, but the default function is a Kaiser function with parameters chosen
to closely mimic the spheroidal function.  Both the Kaiser function and its Fourier transform are easily calculated (in contrast
to the spheroidal function, which requires a complicated model fit that is not as accurate). 

This code also can weight the gridded data by the inverse of the number of nearby data points, which should remove
some of the artifacts due to uneven sampling (see Schwab 1978).  In VLA parlance that is uniform weighting
instead of natural weighting.

The main input parameters are:

| parameter | description |
| :--- | :--- |
| data    | `[n]` Array to be transformed.  If `axis` is specified, this can also be a `[n,:]` or `[:,n]` array (for axis=0 or 1) of time-sampled spectra. |
| time    | `[n]` Times of data samples |
| m       | Number of frequency points in transform |
| return_window | Boolean indicated that the window function should be returned.  This is equivalent to the "dirty beam" in VLA parlance. |

See the function's help for information on other parameters.  The function returns a complex array with `mpts` points (or a 2-D array `[:,mpts]` if the
`axis` parameter if a 2-D data array is used).  If `return_window` is True, a second array (with twice as many points, i.e., `[2*mpts]`) is returned with the window function.

## Usage: autocorr2d, 2-photon autocorrelation function

See White, Sparks, et al. (in preparation) for details of the purpose of this function.  It relies on the same underlying principles as the FDFT, which is why they are bundled together in this package.

## References

Schwab, F. R. 1978, VLA Scientific Memorandum No. 129, "Suppression
of Aliasing by Convolutional Gridding Schemes"

Greisen, E. W. 1979, VLA Scientific Memorandum No. 131, "The Effects
of Various Convolving Functions on Aliasing and Relative 
Signal-to-Noise Ratios"

Schwab, F. R. 1980, VLA Scientific Memorandum No. 132, "Optimal 
Gridding"

## Author

Rick White, 2026 April 2
