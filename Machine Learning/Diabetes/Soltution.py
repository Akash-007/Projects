import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, precision_score, accuracy_score
path ='F:\\Artificial Intelligence\\SimpliLearn\\Datasets'
raw_data = pd.read_csv(path + '\\'+ 'diabetes.csv')
# First Split the data in to three parts Train - Test - Val
# getting the Validation Dataset
n = 100
index_list = random.sample(range(0, raw_data.shape[0] - 1), 100)
x_val = pd.DataFrame()
for i in index_list:
    x_val = x_val.append(raw_data.loc[i])
raw_data.drop(raw_data.index[index_list], inplace=True)
raw_data.reset_index(drop=True, inplace=True)
x_val.reset_index(drop=True, inplace=True)
y_val = x_val['Outcome']
x_val.drop('Outcome', axis=1, inplace=True)
colm = list(raw_data.columns[0:8])
X = raw_data[colm]
Y = raw_data['Outcome']
Y.astype(int)
y_val = y_val.astype(int)

Y = raw_data['Outcome']
raw_data.drop(['Outcome'], axis=1, inplace=True)
print(raw_data.shape)
print(Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#print(x_train.shape)
#print(y_train.shape)
x_test.reset_index(drop=True, inplace=True)
#print(x_test.shape)
#print(x_test.shape)
#print(y_test)
#print(x_test)

model1 = DecisionTreeClassifier()
model1.fit(x_train, y_train)
val_pred = model1.predict(x_val)
test_pred = model1.predict(x_test)
print('Model 1 Accuracy for Test Dataset is {}'.format(accuracy_score(y_test, test_pred)))
print('Model 1 Accuracy for Validation Dataset is {}'.format(accuracy_score(y_val, val_pred)))
val_pred_1 = pd.DataFrame(val_pred)
test_pred_1 = pd.DataFrame(test_pred)

model2 = KNeighborsClassifier(n_neighbors=10)
model2.fit(x_train, y_train)
val_pred_2 = model2.predict(x_val)
test_pred_2 = model2.predict(x_test)
print('Model 2 Accuracy for Test Dataset is {}'.format(accuracy_score(y_test, test_pred_2)))
print('Model 2 Accuracy for Validation Dataset is {}'.format(accuracy_score(y_val, val_pred_2)))
val_pred_2 = pd.DataFrame(val_pred_2)
test_pred_2 = pd.DataFrame(test_pred_2)

meta_features_as_validation = pd.concat([x_val, val_pred_1, val_pred_2], axis=1)
meta_features_as_test = pd.concat([x_test, test_pred_1, test_pred_2], axis=1)

model3 = LogisticRegression(solver='lbfgs', max_iter=3000)
model3.fit(meta_features_as_validation, y_val)
mod_3 = model3.predict(meta_features_as_test)
print('Model 3 Accuracy for Tesing Dataset is {}'.format(accuracy_score(mod_3, y_test)))

