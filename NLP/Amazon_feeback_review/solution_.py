# NLP Project Amazon - Lenovo Review and Topic Modelling
import pandas as pd
from NLP.Projects.twitter import support_file


raw_dataset = pd.read_csv('review.csv')
new_dataset = pd.DataFrame(raw_dataset['review'])
print(new_dataset)


