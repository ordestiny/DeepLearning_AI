# Load Packages
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import sys
import io

def build_data(text, Tx = 40, stride = 3):
    """
    Create a training set by scanning a window of size Tx over the text corpus, with stride 3.

    生成训练集：以stride步长扫描文本，取Tx长度的字符段做为样本X(特征)，X的最后一个字符作为样本Y(Label)

    Arguments:
    text -- string, corpus of Shakespearian poem   诗集
    Tx -- sequence length, number of time-steps (or characters) in one training example  一个训练样本的字符长度
    stride -- how much the window shifts itself while scanning  扫描时窗口移动步长
    
    Returns:
    X -- list of training examples
    Y -- list of training labels
    """
    
    X = []
    Y = []

    ### START CODE HERE ### (≈ 3 lines)
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])
    ### END CODE HERE ###
    
    print('number of training examples:', len(X))
    
    return X, Y


def vectorization(X, Y, n_x, char_indices, Tx = 40):
    """
    Convert X and Y (lists) into arrays to be given to a recurrent neural network.

    将X、Y 向量化

    Arguments:
    X -- 
    Y -- 
    Tx -- integer, sequence length
    
    Returns:
    x -- array of shape (m, Tx, len(chars))
    y -- array of shape (m, len(chars))
    """
    
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.bool)
    y = np.zeros((m, n_x), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
        
    return x, y 

# Theano 中的sample方法
# e^(log(a)) = a
# 当temperature = 1 时，输出原来的概率分布
# 当temperature < 1 时，输出较保守，相当对概率分布做了乘方，原来概率大的会更大，小的会更小，概率分布更陡。
# 当temperature > 1 时，输出较开放，相当对概率分布做了开方，原来概率大会变小，小的会变大，概率分布趁于平缓，这样会增加更多的多样性 。
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # 生成多项式分布
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p = probas.ravel()) #根据概率大小挑选
    return out
    #return np.argmax(probas)
    
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    None
    #start_index = random.randint(0, len(text) - Tx - 1)
    
    #generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    #usr_input = input("Write the beginning of your poem, the Shakespearian machine will complete it.")
    # zero pad the sentence to Tx characters.
    #sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    #generated += sentence
#
    #sys.stdout.write(usr_input)

    #for i in range(400):
"""
        #x_pred = np.zeros((1, Tx, len(chars)))

        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature = 1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
        
        if next_char == '\n':
            continue
        
    # Stop at the end of a line (4 lines)
    print()
 """   
print("Loading text data...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
#print('corpus length:', len(text))

Tx = 40
# 由样本，得到字符表
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
#print('number of unique characters in the corpus:', len(chars))

print("Creating training set...")
X, Y = build_data(text, Tx, stride = 3)
print("Vectorizing training set...")
x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices) 
print("Loading model...")
model = load_model('models/model_shakespeare_kiank_350_epoch.h5')


def generate_output():
    generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    # zero pad the sentence to Tx characters.
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()  # 如果输入的字符数，不足Tx个，前面以0补上
    generated += usr_input 

    sys.stdout.write("\n\nHere is your poem: \n\n") 
    sys.stdout.write(usr_input)
    for i in range(400):

        x_pred = np.zeros((1, Tx, len(chars)))

        # 将已有的句子转为one-host向量
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        # 预测下一个字符的概率
        preds = model.predict(x_pred, verbose=0)[0]
        # 根据概率分布进行采样
        next_index = sample(preds, temperature = 1.0)
        # 转为字符串
        next_char = indices_char[next_index]

        # 更新生成的序列
        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue