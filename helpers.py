import numpy as np

def tilde(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
    
def are_radians_close(a, b, tolerance=1e7):
    return abs(a % np.pi - b % np.pi)/np.pi <= tolerance

round_radians = lambda x: x % (np.pi*2)