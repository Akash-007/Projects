import numpy as np, pandas as pd
import re, random, os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from NLP.Projects.twitter import support_file
import gensim
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.parsing.preprocessing import strip_numeric, strip_short, strip_multiple_whitespaces,strip_non_alphanum,strip_punctuation,strip_tags,preprocess_string
from gensim import models
from gensim import corpora
from collections import Counter
from gensim.models import CoherenceModel


review_tokens = []
clean_review = []
noun_tokens_dict = {}
noun_tokens_list = []

# reading the data
raw_data = pd.read_csv('Tap Reviews.csv')
#print(raw_data.columns)
#print(raw_data['reviews.text'])
# putting the reviews in the list for easy processing
reviews = raw_data['reviews.text'].values
# print(reviews)
for sent in reviews:
    clean_review.append(support_file.simple_cleaning(sent))
# print(clean_review)
for sent in clean_review:
    for i, j in (nltk.pos_tag(word_tokenize(sent))):
        if str(j).startswith('NN'):
            if str(i) in noun_tokens_dict.keys():
                noun_tokens_dict[str.lower(i)] += 1
            else:
                noun_tokens_dict[str.lower(i)] = 1

for i in noun_tokens_dict.keys():
    noun_tokens_list.append([i])
dictionary = corpora.Dictionary(noun_tokens_list)
'''
for i, j in dictionary.items():
    print(str(i) + ' : ' + str(j))
'''
corpus = [word_tokenize(tokens) for tokens in clean_review]
doc_term_matrix = [dictionary.doc2bow(word_tokenize(tokens)) for tokens in clean_review]
tfidf = models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]
result = pd.DataFrame()
print('LSA Model')
lda = models.LdaModel(corpus_tfidf, num_topics=20, id2word=dictionary)
topics = lda.print_topics(num_words=10)
for i in topics:
    print(i[0])
    topi = []
    for stu in word_tokenize(i[1]):
        if str.isalpha(stu):
            topi.append(stu)
    print(topi)
    result[i] = topi
#coher_lda = CoherenceModel(model=lda, texts=corpus, dictionary=dictionary, coherence='c_v')
#index = coher_lda.get_coherence()
#print(index)
result.to_excel('result.xlsx')
