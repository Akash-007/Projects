from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses

import pandas as pd
import numpy as np
raw_data = pd.read_csv(r'C:\Users\HP\PycharmProjects\Learn\Kaggle\m_nist\train\train.csv')
Y = raw_data['label']
raw_data.drop('label', axis=1, inplace=True)
X = np.array(raw_data)
X = X.reshape(X.shape[0], 28, 28, 1)
X = X/255
Y = to_categorical(Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

model_mnist = models.Sequential()
model_mnist.add(layers.Conv2D(32,kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model_mnist.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_mnist.add(layers.Conv2D(24, kernel_size=(2, 2), activation='relu'))
model_mnist.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
model_mnist.add(layers.Flatten())
model_mnist.add(layers.Dense(128, activation='relu'))
model_mnist.add(layers.Dropout(0.3))
model_mnist.add(layers.Dense(24, activation='relu'))
model_mnist.add(layers.Dropout(0.2))
model_mnist.add(layers.Dense(10, activation='softmax'))
model_mnist.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])
model_mnist.fit(x_train, y_train, epochs=10,batch_size=100)
model_mnist.evaluate(x_test,y_test)

This Model is 98.98 % accurate
