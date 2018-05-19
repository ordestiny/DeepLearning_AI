import numpy as np

from deeplearning.con_neural_networks.pad.zero_pad import *

# 一次卷积运算,对应元素相乘，最后累加。
# （数学上卷积需要做一次旋转，但从参数优化的角度，可以不需要这一步）
def conv_single_step(a_slice_prev,W,b):
    s = np.multiply(a_slice_prev,W) + b
    z = np.sum(s)
    return z


# 滑动窗口计算卷积
def conv_slide_windows(A_prev,W,b,stride):
    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    n_H = int(1+ (n_H_prev -f) / stride)
    n_W = int(1 + (n_W_prev -f) / stride)
    Z = np.zeros((n_H,n_W,n_C))

    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                a_slide_prev = A_prev[vert_start:vert_end, horiz_start:horiz_end, :]
                Z[h, w, c] = conv_single_step(a_slide_prev, W[:, :, :, c], b[:, :, :, c])
    return Z


# 卷积前向传播
def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    (f,f,n_C_prev,n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = 1 + int((n_H_prev + 2 * pad -f) / stride)
    n_W = 1 + int((n_W_prev + 2 * pad -f) / stride)

    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slide_prev = A_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i,h,w,c] = conv_single_step(a_slide_prev,W[:,:,:,c],b[:,:,:,c])
    cache = (A_prev,W,b,hparameters)
    return Z,cache

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    return dA_prev, dW, db