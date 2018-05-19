import numpy as np

from deeplearning.con_neural_networks.poolinglayer.pool import *

np.random.seed(1)
A_prev  = np.random.randn(2,4,4,3)

hparameters = {"stride":1,"f":4}

a,cache = pool_forward(A_prev,hparameters)

print("A:",a)

np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print("x = ",x)
print("mask = ",mask)

a  = distribute_value(2,(2,2))

print(a)

np.random.seed(1)

A_prev  = np.random.randn(5,5,3,2)
hparameters = {"stride":1,"f":2}
A,cache  = pool_forward(A_prev,hparameters)

dA = np.random.randn(5,4,2,2)

dA_prev =  pool_backward(dA,cache,mode="max")

print("mean of da = ",np.mean(dA))
print("dA_prev[1,1]",dA_prev[1,1])