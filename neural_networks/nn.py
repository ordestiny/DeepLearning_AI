import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image

# ----------------------------- 算法部分 -------------------------------------------

'''
两个常用的激活函数，一般：隐藏的层使用relu，输出层使用sigmoid
'''
def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return np.maximum(0,Z)

'''
激活函数，前向计算

缓存 Z值
'''
def sigmoid_forward(Z):
	cache =Z
	A = sigmoid(Z)
	return A, cache

def relu_forward(Z):
	cache = Z
	A = relu(Z)
	return A, cache

'''
激活函数，求导

根据 缓存的Z值，计算导数值
'''
def sigmoid_backwark(dA,cache):
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	return dZ

def relu_backwark(dA,cache):
	Z = cache
	dZ = np.array(dA,copy = True)
	dZ[Z<=0] = 0
	return dZ


'''
初始化参数W,b

测试发现，W如果是归一化系数设为1/0.01时，,收敛得很慢
'''
def initialize_parameters_deep(layer_dims):
	np.random.seed(1)
	parameters = {}
	L = len(layer_dims) # 层数
	for l in range(1,L):
		parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
	return parameters

def linear_forward(A,W,b):
	cache = (A,W,b)
	Z  = np.dot(W,A) + b
	return Z,cache

def linear_activation_forward(A,W,b,activation):
	Z,linear_cache = linear_forward(A,W,b)

	if activation == "sigmoid":
		A,activation_cache = sigmoid_forward(Z)
	elif activation == "relu":
		A,activation_cache = relu_forward(Z)
	return A,(linear_cache,activation_cache)

def L_model_forward(X,parameters):
 	caches = []
 	A = X
 	L = len(parameters) //2  # 参数总数除以2，层数
 	for l in range(1,L):
 		A_prev = A
 		A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation="relu")
 		caches.append(cache)
 	AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation="sigmoid")
 	caches.append(cache)
 	return AL,caches # 缓存了每一层的（A,w,b,Z）

def linear_backward(dZ,cache):
	A_prev,W,b = cache
	m = A_prev.shape[1]
	dW = 1./m * np.dot(dZ,A_prev.T)
	db = 1./m * np.sum(dZ,axis = 1,keepdims = True)
	dA_prev = np.dot(W.T,dZ)
	return dA_prev,dW,db


def linear_activation_backward(dA,cache,activation):
	linear_cache ,activation_cache = cache
	if activation == "sigmoid":
		dZ = sigmoid_backwark(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)
	elif activation == "relu":
		dZ = relu_backwark(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	return dA_prev,dW,db

def L_model_backward(AL,Y,cache):

	grads = {}
	L = len(cache)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)
	dA = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
	current_cache = cache[L-1]
	grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dA,current_cache,"sigmoid")
	for l in reversed(range(L-1)):
		current_cache = cache[l]
		grads["dA"+str(l+1)],grads["dW"+str(l+1)],grads["db"+str(l+1)] = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
	return grads
		
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


# 神经网络算法主体
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []                         # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X,parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
 
    return parameters


# ----------------------------- 业务部分 -------------------------------------------

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 加载数据
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# 神经网络的各层神经元个数
layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)