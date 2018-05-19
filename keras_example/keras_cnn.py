import numpy as np

import tensorflow as tf

from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten
from keras.layers import Conv2D,AveragePooling2D,MaxPool2D,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import keras.backend as K

from keras_example.cnn_utils import *

import pydot
import scipy.misc
from matplotlib.pyplot import imshow

K.set_image_data_format("channels_last")
K.set_learning_phase(1)


# 恒等残差块，ResNets中使用的标准块，对应于输入激活（例如 a [1]）与输出激活具有相同维度（例如a [l +2]）。
def identity_block(X,f,filters,stage,block):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block +"_branch"

    F1,F2,F3 = filters

    X_shortcut = X

    # 第一层
    # padding : valid 当最后一次窗口移动， 剩余的元素少于步长时，舍弃。same:当剩余元素少于步长时，边缘补0
    #实际中，考虑计算的成本，对残差块做了计算优化，即将两个3x3的卷积层替换为1x1 + 3x3 + 1x1, 如下图。新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",name= conv_name_base +"2a",\
               kernel_initializer=glorot_uniform(seed=0))(X)
    # 训练神经网络的过程中，每一层的 params是不断更新的，由于params的更新会导致下一层输入的分布情况发生改变，所以进行权重初始化，减小学习率
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X= Activation("relu")(X)

    # 第二层
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",name=conv_name_base+"2b",\
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    # 第三层
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",name=conv_name_base+"2c",\
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)

    return X

# 卷积残差块，当输入和输出尺寸不匹配时，可以使用这种类型的块
def conv_block(X,f,filters,stage,block,s =2):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base  = "bn" + str(stage) + block + "_branch"

    F1,F2,F3 = filters

    X_shortcut = X

    X = Conv2D(F1,kernel_size=(1,1),strides=(s,s),padding="valid" ,name=conv_name_base +"2a",\
               kernel_initializer=glorot_uniform(seed = 0))(X)
    X= BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)

    X = Conv2D(F2,kernel_size=(f,f),strides=(1,1),padding="same",name=conv_name_base+"2b",\
               kernel_initializer=glorot_uniform(seed=0))(X)
    X= BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X= Activation("relu")(X)

    X = Conv2D(F3,kernel_size=(1,1),strides=(1,1),padding="valid",name=conv_name_base+"2c",\
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    X_shortcut = Conv2D(F3,kernel_size=(1,1),strides=(s,s),padding="valid",name=conv_name_base+"1",\
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)

    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
    return X


def ResNet50(input_shape = (64,64,3),classes = 6):
     X_input = Input(input_shape)
     X = ZeroPadding2D((3,3))(X_input) # (70,70,3)

     X = Conv2D(64,(7,7),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X) # (32,32,64)
     X = BatchNormalization(axis=3,name="bn_conv1")(X)
     X = Activation("relu")(X)
     X = MaxPool2D((3,3),strides=(2,2))(X) #(14,14,64)

     X = conv_block(X,f=3,filters=[128,128,512],block="a",s=2,stage=3)
     X = identity_block(X,f=3,filters=[128,128,512],block="b",stage=3)
     X = identity_block(X,f=3,filters=[128,128,512],block="c",stage=3)
     X = identity_block(X,f=3,filters=[128,128,512],block="d",stage=3)

     X = conv_block(X, f=3, filters=[256,256,1024], block="a", s=2,stage=4)
     X = identity_block(X, f=3, filters=[256,256,1024], block="b", stage=4)
     X = identity_block(X, f=3, filters=[256,256,1024], block="c", stage=4)
     X = identity_block(X, f=3, filters=[256,256,1024], block="d", stage=4)
     X = identity_block(X, f=3, filters=[256,256,1024], block="e", stage=4)
     X = identity_block(X, f=3, filters=[256,256,1024], block="f", stage=4)

     X = conv_block(X,f = 3,filters=[512,512,2048],stage=5,block="a")

     X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="b")
     X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="c")

     X = AveragePooling2D(pool_size=(2,2))(X)

     X = Flatten()(X)
     X = Dense(classes,activation="softmax",name="fc" +str(classes),kernel_initializer=glorot_uniform(seed=0))(X)

     model = Model(inputs=X_input,outputs=X,name="ResNet50")

     return model

model = ResNet50(input_shape=(64,64,3),classes=6)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes  = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train =  convert_to_one_hot(Y_train_orig,6).T
Y_test = convert_to_one_hot(Y_test_orig,6).T

model.fit(X_train,Y_train,epochs=20,batch_size=32)

preds = model.evaluate(X_test,Y_test)
print("loss:"+ str(preds[0]))
print("Test Accuracy:" + str(preds[1]))

