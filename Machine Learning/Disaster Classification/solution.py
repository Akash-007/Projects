import pandas as pd
import numpy as np
import re
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pre_processing
from sklearn.metrics import accuracy_score,classification_report


def lists_in_list(data):
    result = []
    for i in data:
        for j in i:
            if j is np.nan:
                pass
            else:
                result.append(j)
    return result


def extract_hash_at(text_data):
    return re.findall('\#\w+', text_data)


def atthe_extract(text_data):
    return re.findall('\@\w+', text_data)


def custom_prepocess_ing(data_fram):
    data = pd.DataFrame(data_fram)
    data['text'] = data['text'].apply(lambda x: re.sub('\w+:\/\/[a-z].[a-z]+\/\w+', '', x))  # http://t.co/o9qknbfOFX
    data['text'] = data['text'].apply(lambda x: re.sub('\d+', '', x))
    data['text'] = data['text'].apply(lambda x: re.sub('[a-z]+\d+', '', x))
    data['text'] = data['text'].apply(lambda x: re.sub('\@\w+', '', x))
    data['cleaned_tweets'] = data['text'].apply(lambda x: pre_processing.simple_cleaning(x, lem=True, tweet=True))
    data['cleaned_tweets'] = data['cleaned_tweets'].apply(lambda x: re.sub('\s{2,}', '', x))
    return data


raw_data = pd.read_csv('train.csv')
raw_data['hash_tags'] = raw_data['text'].apply(lambda x: extract_hash_at(x))
raw_data['atthe_tags'] = raw_data['text'].apply(lambda x: atthe_extract(x))
hash_for_1 = lists_in_list(raw_data.loc[raw_data['target'] == 1, 'hash_tags'].values)
hash_for_0 = lists_in_list(raw_data.loc[raw_data['target'] == 0, 'hash_tags'].values)
atthe_for_1 =lists_in_list(raw_data.loc[raw_data['target'] == 1, 'atthe_tags'].values)
atthe_for_0 =lists_in_list(raw_data.loc[raw_data['target'] == 0, 'atthe_tags'].values)
pre_processing_structure = custom_prepocess_ing(raw_data)

process_data = pre_processing_structure.loc[:, ['cleaned_tweets', 'target']]
Y = process_data['target']
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(process_data['cleaned_tweets'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = Perceptron(max_iter=10000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Testing Accuracy Perceptron {}'.format(accuracy_score(y_pred, y_test)*100))
print(classification_report(y_pred, y_test))

model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print('Testing Accuracy Decision Tree {}'.format(accuracy_score(y_pred2, y_test)*100))
print(classification_report(y_pred2, y_test))

model3 = RandomForestClassifier(n_estimators=150)
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
print('Testing Accuracy RandomForest {}'.format(accuracy_score(y_pred3, y_test)*100))
print(classification_report(y_pred3, y_test))

model4 = LogisticRegression(solver='lbfgs', max_iter=100)
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
print('Testing Accuracy Logistic Regression {}'.format(accuracy_score(y_pred4, y_test)*100))
print(classification_report(y_pred4, y_test))

model5 = KNeighborsClassifier(n_neighbors=50)
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
print('Testing Accuracy KNN {}'.format(accuracy_score(y_pred5, y_test)*100))
print(classification_report(y_pred5, y_test))

testing_data = pd.read_csv('test.csv')
pre_processing_testing = custom_prepocess_ing(testing_data)
X = tfidf.transform(pre_processing_testing['cleaned_tweets'])
y_pred_3 = model5.predict(X)
testing_data['Target'] = y_pred_3
testing_data.to_excel('submit_knn.xlsx')
