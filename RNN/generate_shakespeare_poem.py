from keras.callbacks import LambdaCallback
from keras.models import Model,load_model,Sequential
from keras.layers import Dense,Activation,Dropout,Input,Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io


# 回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
# 通过传递回调函数列表到模型的.fit()中，即可在给定的训练阶段调用该函数集中的函数。
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x,y,batch_size=128,epochs=1,callbacks=[print_callback])