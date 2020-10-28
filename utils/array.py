import numpy as np

ET12 = 2**(1/12)

def norm(x, y, n=2):
    return (x**n + y**n)**(1/n)

def cyclic_padding(array, axis):
    s = [ slice(None) for i in range(array.ndim) ]
    s[axis] = slice(None, -1) 
    return np.concatenate( [ array, array[s] ], axis=axis)

def create_latents(n, max_period, height, width, 
                   quadratic_enhance=True, gamma=ET12, min_=12):
    """
    Returns cos & sin arrays with different periods & directions.
    
    Output shape = 2 x (periods, angles, height, width)

    # Arguments

    n: int, the dimensionality of the direction

    max_period: int/float, maximum period

    quadratic_enhance: bool, default True

    gamma: float, default 1.05946
        growth rate of frequencies
    min_: int, default 12, is related to minimum period
        minimum period = gamma ^ min_ (= 2 by default)
    """
    m = int( np.log2(max_period) / np.log2(gamma) )
    # periods gamma^m ~ gamma^min_
    freqs = gamma**np.arange(-m, -min_+1).reshape(-1, 1, 1, 1)
    f = 2* np.pi* freqs
    theta = np.linspace(0, np.pi, n, endpoint=False).reshape(1, -1, 1, 1)
    y = np.arange(height).reshape(1, 1, -1, 1)
    x = np.arange(width).reshape(1, 1, 1, -1)
    A = f* ( np.cos(theta)* x + np.sin(theta)* y )

    if quadratic_enhance == False:
        return np.cos(A), np.sin(A)
    else:
        enhance_coef = gamma**( 2* np.arange(A.shape[0]) ).reshape(-1, 1, 1, 1)
        return enhance_coef* np.cos(A), enhance_coef* np.sin(A)