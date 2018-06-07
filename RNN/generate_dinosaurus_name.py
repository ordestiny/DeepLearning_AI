import numpy as np
import random
from random import shuffle

from  utils import *

data = open('dinos.txt','r').read()
data = data.lower()
chars = list(set(data))
data_size,vocab_size = len(data),len(chars)

char_to_ix  =  {ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}
print(char_to_ix)
print(ix_to_char)

# 梯度裁剪：将梯度限制在一定范围内，防止梯度爆炸
def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[ 'dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -1 * maxValue, maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients

# 采样生成序列：模型训练完成后，文本生成时，通过对序列的上一个节点进行采样，预测序列下一个节点的值，依此类推，得到最终的结果
def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']  # 结束标志所在的索引，采样时当得到该标志时表示生成结束

    while (idx != newline_character and counter != 50):
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)

        np.random.seed(counter + seed)

        idx = np.random.choice(range(vocab_size), p=y.ravel())
        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a

        seed += 1
        counter += 1

    # 若字符数达到，追加结束标志
    if (counter == 50):
        indices.append(newline_character)

    return indices

def optimize(X,Y,a_prev,parameters,learning_rate = 0.01):
    loss,cache = rnn_forward(X,Y,a_prev,parameters)
    gradients,a =  rnn_backward(X,Y,parameters,cache)
    gradients = clip(gradients, 5)
    prameters = update_parameters(parameters,gradients,learning_rate)
    return loss,gradients,a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):

    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_a, n_x, n_y)

    loss = get_initial_loss(vocab_size, dino_names)

    with open("dinos.txt") as f:
        examples = f.readlines()
    # strip() 移除字符串头尾指定的字符（默认为空格或换行符)
    examples = [x.lower().strip() for x in examples]

    #将序列的所有元素随机排序
    shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        index = j % len(examples)
        # 将样本中的一个字符串，转为数值数组（便于计算）
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        loss = smooth(loss, curr_loss)

        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
            print('\n')
    return parameters

parameters = model(data,ix_to_char,char_to_ix)