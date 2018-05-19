from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.utils import to_categorical

import numpy as np

X_train = np.random.random((1000,20))
Y_train = to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10) # 若为二分类;Y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.random((100,20))
Y_test = to_categorical(np.random.randint(10,size=(100,1)),num_classes=10) #若为二分类 Y_test = np.random.randint(2, size=(1000, 1))

model = Sequential()

model.add(Dense(64,activation="relu",input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))  # 若为二分类：model.add(Dense(10,activation="sigmoid"))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])

model.fit(X_train,Y_train,epochs=20,batch_size=128)
score = model.evaluate(X_test,Y_test,batch_size=128)

print(score)



