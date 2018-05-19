import numpy as np

from deeplearning.con_neural_networks.convlayer.conv import *

np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)
z = conv_single_step(a_slice_prev,W,b)
print(z)

print("--------conv_forward--------")
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b  = np.random.randn(1,1,1,8)
hparameters  = {"pad":2,"stride":1}
Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
print("z's mean:",np.mean(Z))
print("cache_conv:",cache_conv[0][1][2][3])

print("--------conv_backworkd--------")
np.random.seed(1)
dA,dW,db = conv_backward(Z,cache_conv)
print("dA_mean:",np.mean(dA))
print("dW_mean:",np.mean(dW))
print("db_mean:",np.mean(db))