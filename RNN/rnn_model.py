import numpy as np
from rnn_utils import *

def rnn_cell_forward(xt,a_prev,parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Waa,a_prev) + np.dot(Wax,xt) + ba)
    yt_pred = softmax(np.dot(Wya,a_next) + by)
    cache = (a_next,a_prev,xt,parameters)
    return a_next,yt_pred,cache

def rnn_forward(x,a0,parameters):
    caches =[]
    n_x,m,T_x= x.shape
    n_y,n_a =  parameters["Wya"].shape
    a = np.zeros([n_a,m,T_x])
    y_pred = np.zeros([n_y,m,T_x])

    a_next = a0
    for t in range(T_x):
        a_next,yt_pred,cache = rnn_cell_forward(x[:,:,t], a_next,parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    caches = (caches,x)
    return a,yt_pred,caches


def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    # 先把之前存的参数提取出来
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    # 根据  提供的参数 da_next -- Gradient of loss with respect to next hidden state
    # 以及 上面提到的公式 tanh(u) = (1- tanh(u)**2)*du ，这里的 du 就是 da_next  tanh(u) 是 a_next
    dtanh = (1 - a_next **2)* da_next
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    #  axis=0 列方向上操作 axis=1 行方向上操作  keepdims=True 矩阵的二维特性
    dba = np.sum(dtanh, axis=1, keepdims=True)
    gradients = {"dxt":dxt, "da_prev":da_prev,"dWax":dWax, "dWaa":dWaa, "dba":dba }
    return gradients

def rnn_backward(da,caches):
    (caches,x) = caches
    (a1,a0,x1,parameters) =  caches[0]

    n_a,m,T_x = da.shape
    n_x,m = x1.shape

    dx = np.zeros([n_x,m,T_x])
    dWax = np.zeros([n_a,n_x])
    dWaa = np.zeros([n_a,n_a])
    dba = np.zeros([n_a,1])
    da0 = np.zeros([n_a,m])
    da_prevt = np.zeros([n_a,m])

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t] + da_prevt,caches[t])
        dxt,da_prevt,dWaxt,dWaat,dbat = gradients["dxt"],gradients["da_prev"],gradients["dWax"],gradients["dWaa"],gradients["dba"]
        dx[:,:,t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba  += dbat

    da0 = da_prevt
    gradients = {"dx":dx, "da0":da0,"dWax":dWax, "dWaa":dWaa, "dba":dba }
    return gradients
