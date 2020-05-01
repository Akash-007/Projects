import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def create_dataset(data_x):
    df_x = []
    df_y = []
    for i in range(len(data_x)):
        if i == (len(data_x) - 1):
            pass
        else:
            df_x.append(data_x[i])
            df_y.append(data_x[i + 1])
    return np.array(df_x), np.array(df_y)


def predict_items(model1, firstVale, length):
    predict_items = []
    predict_items.append(firstVale)
    for i in range(length):
        predi_val = model1.predict(np.reshape(predict_items[-1], (1, 1, 1)))
        predict_items.append(predi_val)
        print(predi_val)
    return predict_items


raw_data = pd.read_excel('prices.xlsx')
#print(raw_data)
stock_prices = pd.DataFrame(raw_data['close'])
scaler = MinMaxScaler(feature_range=(0,1))
stock_prices = scaler.fit_transform(stock_prices)
#print(len(stock_prices))
train_len = int(len(stock_prices)*(0.8))
test_len = len(stock_prices) - train_len
train_data = stock_prices[0:train_len]
test_data = stock_prices[train_len:]
X_train, Y_train = create_dataset(train_data)
X_test, Y_test = create_dataset(test_data)
#print(X_train.shape[0])
#print(X_train.shape[1])
X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))
#print(np.reshape(range(0,27), (27,1,1)))

#print(X_test[0:2])
#print(Y_test[0:2])
#print(X_train[0:2])
#print(Y_train[0:2])
#plt.plot(X_test)
#plt.show()
#building Model

model = models.Sequential()
model.add(layers.LSTM(50, input_dim=1, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer='Adam', loss='mse')
model.fit(X_train, Y_train, batch_size=20, epochs=10)
print(model.evaluate(X_test, Y_test))
result = predict_items(model, X_test[0], 5)
print(result)
print(Y_test[0:7])
