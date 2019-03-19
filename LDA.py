import nltk
from nltk.corpus import stopwords
import string
import re
import collections
import math

import pandas as pd

###

import os
import numpy as np

from gensim import corpora, models, similarities

import gensim
import logging
import tempfile

from gensim import corpora
###

from nltk.corpus import stopwords
from string import punctuation

import warnings
warnings.filterwarnings("ignore")


# remove common words and tokenize
list1 = ['RT','rt']
stoplist = stopwords.words('english') + list(punctuation) + list1
print ('start \n')
texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in all_text]

print('Done pre process \n')


####

dictionary = corpora.Dictionary(texts)


print ('Done Dictionary. \n')

corpus = [dictionary.doc2bow(text) for text in texts]

print ('corpus don \n')
from gensim import corpora, models, similarities

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

total_topics = 5

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)
corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lda.show_topics(total_topics,5)

from collections import OrderedDict

data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}


df_lda = pd.DataFrame(data_lda)
print(df_lda.shape)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)

import seaborn as sns
import matplotlib.pyplot as plt

g=sns.clustermap(df_lda.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(12, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

    