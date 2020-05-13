import os
import gensim
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from gensim.utils import lemmatize
import nltk
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.parsing.preprocessing import strip_numeric
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim import similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stop_nltk = stopwords.words('english')
vectorizer = TfidfVectorizer(max_features=3000)


def pre_processing():
    for document in texts:
        doc = strip_numeric(stem_text(document))
        yield gensim.utils.tokenize(doc, lower=True)


raw_data = pd.read_table("r8-all-terms.txt", sep="\t", names=['label', 'text'])
model_lsi = LsiModel.load('lis_search.model')
dict = corpora.Dictionary.load('modeldic.dic')
index = similarities.Similarity.load('own_search_engine.index')
doc = 'crude oil price'# Query
vec_bow = dict.doc2bow(doc.lower().split())
vec_lsi = model_lsi[vec_bow]
unsorted_similarity = index[vec_lsi]
sorted_similarity = sorted(enumerate(unsorted_similarity), key=lambda item: -item[1])
tops = 0
print('Displaying top 10 records')
for index, similarity in sorted_similarity:
    if tops <= 10:
        print(similarity, raw_data.loc[index, ['text']])
        tops += 1
    else:
        break
