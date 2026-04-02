import numpy as np
import re
import sys

def printit(label, a, *, ncols=12, width=11, precision=6):
    """print numpy array with ncols column having fixed width and precision
    handles complex arrays and suppresses negative zero
    """
    print(label)
    a = np.asarray(a)
    if str(a.dtype).startswith('complex'):
        a2 = np.zeros(len(a)*2)
        a2[0::2] = a.real
        a2[1::2] = a.imag
    else:
        a2 = a
    sform = "{:>"+str(width)+"}"
    # pattern to remove brackets around array
    bpat = re.compile(r"[\[\]]")
    # suppress=True is claimed to avoid printing negative zero but does not always work
    negzero = re.compile(r"-0\." + ("0"*precision))
    poszero = "0." + ("0"*precision)
    with np.printoptions(suppress=True, precision=precision, floatmode="fixed", linewidth=250):
        for i in range(0, len(a2.flat), ncols):
            s = negzero.sub(poszero, bpat.sub("", np.array_str(a2.flat[i:i+ncols])) )
            print("".join([sform.format(x) for x in s.split()]))
