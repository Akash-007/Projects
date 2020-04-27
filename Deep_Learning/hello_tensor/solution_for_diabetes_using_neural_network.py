try:
    import pandas as pd
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification, accuracy_score
    print('All the support file imported')
except :
    print('File not imported')

# getting the data
path = r'F:\Artificial Intelligence\SimpliLearn\Datasets'
raw_data = pd.read_csv(path + '\\' + 'diabetes.csv')
X = raw_data.iloc[:, 0:8]
Y = raw_data.iloc[:, 8]
statu = 'train'
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


if statu == 'train':
    model = Sequential()
    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=80, batch_size=10)
    _, accuracy = model.evaluate(x_train, y_train)
    print('accuracy of the matrix is {}'.format(accuracy))
    print('Model Trained')
