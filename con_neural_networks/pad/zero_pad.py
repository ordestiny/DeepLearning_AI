import numpy as np
def zero_pad(X,pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),"constant",constant_values=0)
    return X_pad
