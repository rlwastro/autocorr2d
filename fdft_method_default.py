# define default paramters for resampling

import numpy as np
from kaiser import kaiser
from binfunc import binfunc
from sphfn import sphfn

def fdft_method_default(method='kaiser', dim=2, **kw):
    """Define default parameters for a resampling/weighting method.
    
    Usage: dmethod = fdft_method_default([method] [, dim=n])

    method  Specifies the choice of function and some parameters.
            Default method is 'kaiser'.
            Available methods are 'kaiser', 'sphfn', 'binfunc',
                'binfunc1', 'binfunc2', 'binfunc3', 'binfunc4'.
            method can be a string (e.g., 'kaiser') or a dict,
                e.g., dict(name='kaiser', support=8)
            Note that additional parameters included in the method
                dict get propagated to the output results.
    dim     1 or 2 (default 2). dim allows different defaults for 1D
            and 2D transforms (not currently used)

    Function result is a dict including the name of the method and
    other parameters either included in the input method parameter or
    specified in the default function for that method.
    """

    if dim != 1 and dim != 2:
        raise ValueError('dim must be 1 or 2')

    if isinstance(method, str):
        smethod = dict(name=method, **kw)
    elif isinstance(method, dict):
        smethod = method.copy()
        if 'name' not in smethod:
            raise ValueError('method must have a key "name"')
        smethod.update(kw)
    else:
        raise ValueError('method must be a string or a dict')

    funcdict = dict(kaiser=_kaiser_fdft_default,
                    sphfn=_sphfn_fdft_default,
                    binfunc=_binfunc_fdft_default,
                    binfunc1=_binfunc1_fdft_default,
                    binfunc2=_binfunc2_fdft_default,
                    binfunc3=_binfunc3_fdft_default,
                    binfunc4=_binfunc4_fdft_default)
    try:
        finit = funcdict[smethod['name'].lower()]
    except KeyError:
        raise ValueError(f'method name must be one of {" ".join(funcdict.keys())}') from None
    rv = finit(dim, **smethod)
    return rv


def _kaiser_fdft_default(dim, name='kaiser', support=8, beta=4.09*np.pi, function=kaiser, **kw):
    assert name.lower() == 'kaiser'
    rv = dict(name=name, function=function, support=support, beta=beta, **kw)
    return rv

def _sphfn_fdft_default(dim, name='sphfn', support=8, exponent_id=1, function=sphfn, **kw):
    assert name.lower() == 'sphfn'
    rv = dict(name=name, function=function, support=support, exponent_id=exponent_id, **kw)
    return rv

def _binfunc1_fdft_default(dim, name='binfunc1', **kw):
    assert name.lower() == 'binfunc1'
    return _binfunc_fdft_default(dim, name='binfunc', order=1, **kw)

def _binfunc2_fdft_default(dim, name='binfunc2', **kw):
    assert name.lower() == 'binfunc2'
    return _binfunc_fdft_default(dim, name='binfunc', order=2, **kw)

def _binfunc3_fdft_default(dim, name='binfunc3', **kw):
    assert name.lower() == 'binfunc3'
    return _binfunc_fdft_default(dim, name='binfunc', order=3, **kw)

def _binfunc4_fdft_default(dim, name='binfunc4', **kw):
    assert name.lower() == 'binfunc4'
    return _binfunc_fdft_default(dim, name='binfunc', order=4, **kw)

def _binfunc_fdft_default(dim, name='binfunc', support=None, order=None, function=binfunc, **kw):
    assert name.lower() == 'binfunc'
    if order is None:
        raise ValueError('order must be specified')
    if not isinstance(order,int) or order < 1 or order > 4:
        raise ValueError('order must be an integer between 1 and 4')
    if support is None:
        support = order
    rv = dict(name=name, function=function, support=support, order=order, **kw)
    return rv

if __name__ == "__main__":
    for c in ('kaiser', 'binfunc1', 'binfunc2', 'sphfn'):
        print(c, fdft_method_default(c))
    print('kaiser support=3', fdft_method_default('kaiser', support=3))
    try:
        print('junk', fdft_method_default('junk'))
    except ValueError as e:
        if str(e)=="method name must be one of kaiser sphfn binfunc binfunc1 binfunc2 binfunc3 binfunc4":
            print("Got expected failure for missing name 'junk'")
        else:
            raise
