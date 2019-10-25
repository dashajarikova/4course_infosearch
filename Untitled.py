#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask import render_template, request, redirect, url_for

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np 
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2
import re
from nltk.corpus import stopwords 

import warnings
import nltk
from gensim.models.keyedvectors import KeyedVectors

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()

import nltk
nltk.download('stopwords')

import logging
logging.basicConfig(filename='preprocessing.log', 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.INFO)

from gensim.models.keyedvectors import KeyedVectors 
fast_model = 'fasttext/model.model'
 fasttext_model = KeyedVectors.load(fast_model)

def open_data():
    data = pd.read_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')
    
    data = data.drop(['question2', 'is_duplicate'], axis=1)[:100]
    
    data['question1'] = data['question1'].apply(lambda x: preproc(x)) 
    data.to_csv('preprocessed_data.csv', index=True)
    return data

def tf_idf_indexing(d): 
    vec = TfidfVectorizer()
    X = vec.fit(d) 
    df_tfidf = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    #print(X)
    df_tfidf.to_csv('tf_idf_index.csv', index=False)
    
    joblib.dump(vec, 'tf_idf_vectorizer.pkl') #создаем файл пикл, где все переменные
    return df_tfidf

def bm25_indexing(d, k=2, b=0.75): 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(d)
    term_freq_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    term_freq_counts['sum'] = term_freq_counts.sum(axis=1)
    tf_table = term_freq_counts.div(term_freq_counts['sum'], axis=0)
    tf_table = tf_table.fillna(0)    
    tf_table = tf_table.drop(['sum'], axis=1)
    
    bin_vectorizer = CountVectorizer(binary=True)
    bin_X = bin_vectorizer.fit_transform(d)
    bin_counts = pd.DataFrame(bin_X.toarray(), columns=bin_vectorizer.get_feature_names()) 
    word_counter_dict = {}
    for column in bin_counts.columns:
        col = bin_counts[column]
        sum_ = col.sum()
        word_counter_dict[column] = sum_
    inverse_counter = pd.DataFrame.from_dict(word_counter_dict, orient='index')
    inverse_counter = inverse_counter.transpose()
    
    #N = d.shape[0]
    N = len(d)
    idfs = {}
    for w in inverse_counter:
        idf = log((N - inverse_counter[w] + 0.5)/(inverse_counter[w] +0.5))
        idfs[w] = idf
    idf_table = pd.DataFrame.from_dict(idfs, orient='index')
    idf_table = idf_table.transpose()

    sums = term_freq_counts['sum']
    avg = term_freq_counts['sum'].mean()
    sums_normalized = sums.div(avg)

    conversion_table_numerator = tf_table.mul(k+1)
    coefficient = sums_normalized.mul(b)
    coefficient = coefficient.add(1-b)
    coefficient = coefficient.mul(k)
    
    conversion_table_denominator = tf_table.mul(coefficient, axis=0)
    tf_factor = conversion_table_numerator.divide(conversion_table_denominator) 
    tf_factor = tf_factor.fillna(0)
    n = tf_factor.shape[0]
    
    idf_table = pd.concat([idf_table]*n, ignore_index=True)
    bm25_table = tf_factor.mul(idf_table, axis=1)
    bm25_table = bm25_table.fillna(0)
    bm25_table.to_csv('bm25_index.csv', index=False)
    return bm25_table

def getting_fasttext(filepath):
    fasttext_model = KeyedVectors.load(filepath)
    return fasttext_model

def sent_vectorizer(sent, model):
    if type(sent) != str:
        sent_vector = np.zeros((model.vector_size,))
        return sent_vector
    sent = sent.split()
    lemmas_vectors = np.zeros((len(sent), model.vector_size))
    for idx, lemma in enumerate(sent):
        if lemma in model.vocab:
            lemmas_vectors[idx] = model[lemma]
    sent_vector = lemmas_vectors.mean(axis=0)
    return sent_vector

def fasttext_indexing(d):
    model = getting_fasttext('fasttext/model.model')
    vectors_dict = {}
    for idx, row in d.iterrows():
        sent_vec = sent_vectorizer(row.question1, model)
        vectors_dict[idx] = sent_vec
    data = pd.DataFrame.from_dict(vectors_dict, orient='index')
    data.to_csv('fasttext_index.csv', index=False)
    return data

raw_df = open_data()
logging.info('made preprocessed dataframe')
del(raw_df)
preproc_df = preproc_opening()
tf_idf_index = tf_idf_indexing(list(preproc_df.question1))
logging.info('made tf-idf dataframe')
del(tf_idf_index)
bm25_index = bm25_indexing(list(preproc_df.question1))
logging.info('made bm25 dataframe')
del(bm25_index)
fasttext_index = fasttext_indexing(preproc_df)
logging.info('made fasttext dataframe')
del(fasttext_index)

