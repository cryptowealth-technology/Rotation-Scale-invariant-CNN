import numpy as np

def cyclic_padding(array, axis):
    s = [ slice(None) for i in range(array.ndim) ]
    s[axis] = slice(None, -1) 
    return np.concatenate( [ array, array[s] ], axis=axis)