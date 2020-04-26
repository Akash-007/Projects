# Twitter Data Analysis
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from supp_file import extract_keyword, lists_insidelist
from NLP.Projects.twitter import support_file
tfidf = TfidfVectorizer()


def find_match(text_data):
    result = []
    for i in str(text_data).split(' '):
        if i in total_tokens:
            result.append(i)
        else:
            pass
    return ' '.join(result)


proces = True
raw_data = pd.read_csv('train.csv')
raw_data.index = raw_data['textID']
raw_data.drop('textID', axis=1, inplace=True)
raw_data['sentiment'].replace({'neutral': 0, 'positive': 1, 'negative': -1}, inplace=True)

if proces == True:
    result = extract_keyword(raw_data)
    positi = lists_insidelist(result[0])
    negativ =lists_insidelist(result[1])
    neutral = lists_insidelist(result[2])
else:
    pass

#sentiment_data['clean_data'] = sentiment_data.text.apply(lambda x: support_file.simple_cleaning(x, lem=True))
total_tokens = set(positi + negativ + neutral)
testing_dataframe = pd.read_csv('test.csv')
testing_dataframe['words'] = testing_dataframe['text'].apply(lambda x: find_match(x))
testing_dataframe.to_csv('result1.csv')


Y = sentiment_data['sentiment']
X = tfidf.fit_transform(sentiment_data['clean_data'].values.astype(str))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
model1 = DecisionTreeClassifier()
model1.fit(x_train, y_train)
print('Model 1 build Successfully')
y_pred1 = model1.predict(x_test)
print('Accuracy is {}'.format(accuracy_score(y_pred1, y_test)))
model2 = RandomForestClassifier(max_depth=200, n_estimators=300)
model2.fit(x_train, y_train)

print('Model 2 build Successfully')
y_pred2 = model2.predict(x_test)
print('Accuracy is {}'.format(accuracy_score(y_pred2, y_test)))


model3 = LogisticRegression(solver = 'lbfgs')
model3.fit(x_train, y_train)
print('Model 3 build Successfully')
y_pred3 = model3.predict(x_test)
print('Accuracy is {}'.format(accuracy_score(y_pred3, y_test)*100))

model5 = XGBClassifier()
model5.fit(x_train, y_train)
print('Model 5 build Successfully')
y_pred5 = model5.predict(x_test)
print('Accuracy is {}'.format(accuracy_score(y_pred5, y_test)*100))

model4 = KNeighborsClassifier(n_neighbors=30, algorithm='kd_tree')
print('Model 4 build Successfully')
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
print('Accuracy is {}'.format(accuracy_score(y_pred4, y_test)*100))

