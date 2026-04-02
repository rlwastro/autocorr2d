import numpy as np
from fdft_method_default import fdft_method_default
import sys

def convgrid2d(x, y, xmin, ymin, dx, dy, mx, my, wt=None,
               method='kaiser', verbose=False):
    """Convolve data on unequally spaced grid to equally spaced grid
    Convolved data is multiplied by a weighting function, wt, if wt is specified

    Inputs:
    x, y        [n] 1-D arrays with locations
    xmin, ymin  minimum values for grid
    dx, dy      grid spacing
    mx, my      number of points in grid
    wt          [n] 1-D corresponding weights (optional)
    method      Interpolation method.  This must be a string or a dict
                with a field 'name' that identifies the method along with
                whatever additional parameters are desired.  For example, use
                method=dict(name="kaiser", beta=4.09*np.pi, support=8)
                to specify parameters for the Kaiser function.  See the
                fdft_method_default module for the defaults.
    verbose     If True, print some stuff

    Returns [my,mx] array with point locations convolved to the grid.
    """

    dmethod = fdft_method_default(method, dim=2)
    support = dmethod['support']

    try:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-D arrays")
        if x.size != y.size:
            raise ValueError("x and y must be same length")
    except AttributeError as e:
        raise ValueError("x and y must be 1-D numpy arrays")
    n = x.size
    hsize = support/2.0
    ihsize = np.floor(hsize).astype(int)
    if verbose:
        if dmethod['name'] == 'sphfn':
            print(f"convgrid2d: {dmethod['name']} {dmethod['exponent_id']=} {dmethod['support']=} {hsize=} {ihsize=}")
        elif dmethod['name'] == 'kaiser':
            print(f"convgrid2d: {dmethod['name']} {dmethod['beta']=} {dmethod['support']=} {hsize=} {ihsize=}")
        else:
            print(f"convgrid2d: {dmethod['name']} {dmethod['order']=} {dmethod['support']=} {hsize=} {ihsize=}")

    u = (x-xmin)/dx
    v = (y-ymin)/dy

    # loop limits plus round/floor switch work for odd or even support size
    odd = support % 2
    if odd:
        iu = np.round(u).astype(int)
        iv = np.round(v).astype(int)
    else:
        iu = np.floor(u).astype(int)
        iv = np.floor(v).astype(int)
    # compute offsets from bin edges
    u -= iu
    v -= iv

    # accumulate results in 1-D array
    nim = my*mx
    cdata = np.zeros(nim, dtype=float)

    # precompute the ysfn and yindex values
    ysfn = np.empty((support,n),dtype=float)
    yindex = np.empty((support,n),dtype=int)
    for j, l in enumerate(range(-ihsize+1-odd,ihsize+1)):
        ysfn[j] = dmethod['function']((v-l)/hsize, gridding=True, normalize=True, **dmethod)
        yindex[j] = ((iv+l) % my)*mx

    for k in range(-ihsize+1-odd,ihsize+1):
        xsfn = dmethod['function']((u-k)/hsize, gridding=True, normalize=True, **dmethod)
        if wt is not None:
            xsfn *= wt
        xindex = ((iu+k) % mx)
        for j in range(support):
            cdata += np.bincount(xindex+yindex[j], minlength=nim, weights=xsfn*ysfn[j])

    # convert back to 2D array
    return cdata.reshape(my,mx)

if __name__ == "__main__":
    from printit import printit

    if len(sys.argv) <= 1:
        methods = ["kaiser"]
    else:
        methods = sys.argv[1:]

    # create test data for convgrid2d

    n = 987 # fibonacci number
    ### n = 55 # fibonacci number
    # golden ratio spacing is fairly random
    phi = (1.0+np.sqrt(5.0))/2.0
    omega = (np.arange(n)*phi) % 1.0
    r = (((n + np.arange(n))*np.arange(n))*phi) % 1.0
    x = (r*np.sin(2*np.pi*omega) + 1) % 1.0
    y = (r*np.cos(2*np.pi*omega) + 1) % 1.0

    # not needed?
    data = np.sin(5*2*np.pi*x)*np.sin(4*2*np.pi*y)

    xmin = 0.0
    ymin = 0.0
    mx = 64
    my = 64
    dx = 1.0/(mx-1)
    dy = 1.0/(my-1)

    for method in methods:
        cdata = convgrid2d(x, y, xmin, ymin, dx, dy, mx, my, method=method)
        print('convgrid2d',method)
        printit('x', x)
        printit('y', y)
        printit('cdata', cdata)
