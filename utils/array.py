import numpy as np
import math

ET12 = 2**(1/12)

def log_floor(x, base):
    """Return floor of log"""
    return int( math.log(x, base) + 0.001)

def norm(x, y, n=2):
    return (x**n + y**n)**(1/n)

def normalize(A, axis=1):
    return ( A - A.mean(axis, keepdims=True) ) / A.std(axis, keepdims=True)

def cyclic_padding(array, axis):
    s = [ slice(None) for i in range(array.ndim) ]
    s[axis] = slice(None, -1) 
    return np.concatenate( [ array, array[s] ], axis=axis)

def create_latents(n, max_period, height, width, gamma=ET12, min_period=2):
    """
    Returns cos & sin arrays with different periods & directions.
    
    Output shape = 2 x (periods, angles, height, width)

    # Arguments

    n: int, the dimensionality of the direction

    max_period: int/float, maximum period

    gamma: float, default 1.05946
        growth rate of frequencies
    min_: int, default 12, is related to minimum period
        minimum period = gamma ^ min_ (= 2 by default)
    """
    m = log_floor(max_period, gamma)
    n = log_floor(min_period, gamma)
    # periods gamma^m ~ gamma^n
    freqs = gamma**np.arange(-m, -n+1).reshape(-1, 1, 1, 1)
    f = 2* np.pi* freqs
    theta = np.linspace(0, np.pi, n, endpoint=False).reshape(1, -1, 1, 1)
    y = np.arange(height).reshape(1, 1, -1, 1)
    x = np.arange(width).reshape(1, 1, 1, -1)
    A = f* ( np.cos(theta)* x + np.sin(theta)* y )
    return np.cos(A), np.sin(A)

def transformation(img, cos, sin, standardize=False, gamma=ET12, min_period=2):
    """
    Return projections of input image on larent vectors (frequencys, directions).
    
    Shape of cos & sin: (frequency, direction, height, width) 
    """
    h, w = img.shape
    d = log_floor( norm(h, w)/min_period, gamma) # length of diagonal as maximum period of img
    c = norm(
             np.sum( img * cos[-d:, :, :h, :w], axis=(-2, -1) ), 
             np.sum( img * sin[-d:, :, :h, :w], axis=(-2, -1) )
            )
    if standardize == True:
        return normalize(c)
    else:
        return c
