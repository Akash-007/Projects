import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report,recall_score,precision_score

'''
1. Import the three datasets
2. Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. (Hint: (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID & UserId)
3. Explore the datasets using visual representations (graphs or tables), also include your comments on the following:
    3.a User Age Distribution
    3.b User rating of the movie “Toy Story”
    3.c Top 25 movies by viewership rating
    3.d Find the ratings for all the movies reviewed by for a particular user of user id = 2696
4. Feature Engineering:
    Use column genres:
    4.a Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)
    4.b Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre. 
    4.c Determine the features affecting the ratings of any particular movie.
5. Develop an appropriate model to predict the movie rating
'''

movies = pd.read_csv('movies.dat', sep='::', header=None, names=['movie_id', 'movie_name', 'genere'])
# print(movies.shape)
# print(movies.columns)
ratings = pd.read_csv('ratings.dat', sep='::', header=None, names=['user_id','movie_id','rating','timestamp'])
# print(ratings.shape)
# print(ratings.columns)
users = pd.read_csv('users.dat', sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
# print(users.shape)
# print(users.columns)
# merging movie and ratings
movie_rating = pd.merge(movies, ratings, how='inner', on='movie_id')
# print(movie_rating.shape)
# merging movie_rating and users
final_dataset = pd.merge(users, movie_rating, how='inner', on='user_id')
print('Final Dataset Shape : ' + str(final_dataset.shape))
# Step 1 and 2 Completed
# 3.1
plt.figure(figsize=(10, 8))
seaborn.countplot(x=final_dataset['age'])
plt.show()
# analysis 1 : people between age 25 and 35 are more who watch movie
# 3.2
pd.value_counts(final_dataset.loc[final_dataset['movie_name'] == 'Toy Story (1995)', 'rating']).sort_index().plot(kind='bar', title='Rating for Toy Story')
plt.show()
print('Toy Story is rated 4 many times')
# 3.3
aggregation = [np.mean, np.median, np.min, np.max]
print('Top 25 movies according to the rating')
datset = pd.DataFrame(final_dataset.groupby('movie_name').mean()[['rating']])
print(datset.sort_values('rating', ascending=False).head(25))
# 3.4
print('movies rated by the user 2696')
print(final_dataset.loc[final_dataset['user_id'] == 2696, ['user_id', 'movie_name', 'rating']])
# 4.1
gen_dat = pd.DataFrame(final_dataset.groupby('genere').mean()[['rating', 'age']])
gen_dat.sort_values(by=['rating','age'], ascending=False)
print('most rated movies types are Animation|Comedy|Thriller and Sci-Fi|War and rating is  4.5')
# 4.2
# get_dummies create one hot encoding for genere
hot_end = pd.get_dummies(final_dataset['genere'])
new_data_set = pd.concat([final_dataset,hot_end], axis=1)
# 4.3
# as we can notice all the column here is categorical column so visualizing all the plot
cate = ['gender','occupation','genere','age']

for i in cate:
    table = pd.value_counts(final_dataset[i])
    plt.style.use('dark_background')
    plt.pie(x=table, labels=table.index)
    #seaborn.violinplot(x=table)
    plt.show()

print('plotting categorical values')

for i in cate:
    seaborn.countplot(x=i, hue='rating', data=final_dataset)
    plt.show()
print('After visualizing all the variables the imp factor for rating is age and gender')

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
seaborn.countplot(x='age', hue='rating', data=final_dataset)
plt.subplot(1,2,2)
seaborn.countplot(x='gender', hue='rating', data=final_dataset)
plt.show()
# 5
# building the model
#print(final_dataset.columns)
# removing unwanted columns
final_dataset.drop(['user_id','zipcode','movie_id','movie_name','timestamp'], axis=1, inplace=True)
final_dataset['gender'] = final_dataset['gender'].replace({'M': 0, 'F': 1})
final_dataset.drop(['genere'], axis=1, inplace=True)
#new_data = pd.concat([final_dataset,hot_end], axis=1)
#new_data.reset_index()
#new_data = pd.concat([final_dataset,hot_end], axis=1)
#new_dataset = pd.DataFrame(new_data)
Y = pd.DataFrame(final_dataset['rating'])
final_dataset.drop(['rating'], axis=1, inplace=True)
#new_dataset.drop(['rating'], axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(final_dataset, Y, test_size=0.3)
