# this class can be used to do basic operations of text cleansing
# Cleaning of Text Means : 1) Stop Words 2) Case Normalization 3) Remove Punctuations and symbols 4) Stemming and Lemit 5)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.parsing.preprocessing import strip_numeric, strip_short, strip_multiple_whitespaces,strip_non_alphanum,strip_punctuation,strip_tags,preprocess_string
import gensim

lematize = WordNetLemmatizer()


def simple_cleaning(text_data, **lemop):
    text_data = str(text_data)
    lem = lemop.get('lem')
    result = []
    result1 = []
    lemat = []
    puncutations = string.punctuation
    stop_words = stopwords.words('english')
    # lower
    lower_text = text_data.lower()
    # removing symbols and punctuations
    for i in lower_text:
        if i in puncutations:
            pass
        else:
            result.append(i)
    new_string = ''.join(result)
    for i in word_tokenize(new_string):
        if lemop.get('St') is True:
            if i in stop_words:
                pass
            else:
                result1.append(i)
        else:
            result1.append(i)

    if lem is True:
        for token in result1:
            lemat.append(lematize.lemmatize(token))
        return ' '.join(lemat)
    else:
        return ' '.join(result1)


def pre_processing(corpus):
    for document in corpus:
        doc = strip_numeric(document)
        doc = remove_stopwords(doc)
        doc = strip_short(doc, 3)
        #doc = stem_text(doc)
        doc = strip_punctuation(doc)
        strip_tags(doc)
        yield gensim.utils.tokenize(doc, lower=True)
