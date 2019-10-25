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

from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
# X = vect.fit_transform

data = pd.read_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')

fast_model = 'fasttext/model.model'
fasttext_model = KeyedVectors.load(fast_model)

warnings.filterwarnings("ignore")
w2v = getting_fasttext('fasttext/model.model') #ну ок

def preproc(text): 
    morph = pymorphy2.MorphAnalyzer()
    text = re.sub(r'[A-Za-z0-9<>В«В»\.!\(\)?,;:\-\"]', r'', text)
    text = WordPunctTokenizer().tokenize(text)
    stopword_list = set(stopwords.words('russian'))
    
    preproc_text = ''
    for w in text:
        if w not in stopword_list:
            new_w = morph.parse(w)[0].normal_form + ' '
            preproc_text += new_w

    return preproc_text

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

def calc_metric(query, data):
    cos_sim = data.apply(lambda row: cosine_similarity(row.values.reshape(1, -1), query)[0][0], axis=1) #сложна
    cos_sim = pd.DataFrame(cos_sim, columns=['val'])
    best_cos_sim = cos_sim.nlargest(10, 'val')
    return best_cos_sim

def metric_bm25(query):
    df = pd.read_csv('bm25_index.csv', index_col=None)
    query = query.split(' ')
    lemmas_list = list(df.columns)
    query_bm25 = {}
    for lemma in lemmas_list:
        if lemma in query:
            query_bm25[lemma] = [1]
        else:
            query_bm25[lemma] = [0]
    query_bm25 = pd.DataFrame.from_dict(query_bm25)
    metric_value = calc_metric(query_bm25, df)       
    return metric_value

def metric_tf(query):
    df = pd.read_csv('tf_idf_index.csv', index_col=None)
    vectorizer = joblib.load('tf_idf_vectorizer.pkl')
    query_tfidf = vectorizer.transform([query])
    query_tfidf = pd.DataFrame(query_tfidf.toarray(), columns=vectorizer.get_feature_names())
    
    metric_value = calc_metric(query_tfidf, df)
    return metric_value

def metric_fast(query):
    df = pd.read_csv('fasttext_index.csv', index_col=None)
    sent_vector = sent_vectorizer(query, w2v)
    query_fasttext = np.asarray(sent_vector).reshape(1, -1)
    metric = calc_metric(query_fasttext, df)
    return metric

def top_docs(result):
    q_dict ={}
    for idx, row in result.iterrows():
        for id_doc, doc in enumerate (data['question2']):
            if idx==id_doc:
                q_dict[idx] = [doc, row.val]
    return (q_dict)

app = Flask(__name__)

@app.route('/')
def query():
    if request.args:
        query = request.args['user_query']
        type_metrics = request.args['type_metrics']
        if type_metrics == 'BM25':
            values = metric_bm25(query)
            metrics = 'Вы выбрали BM25 и вот посмотрите:'
            top10_doc = top_docs(values)
        elif type_metrics == 'tf':
            values = metric_tf(query,data)
            metrics = 'Вы выбрали TFIDF и вот посмотрите:'
            top10_doc = top_docs(values)
        elif type_metrics == 'fasttext':
            values = metric_fast(query)
            metrics = 'Вы выбрали Fasttext и вот посмотрите:'
            top10_doc = top_docs(values)
        return render_template('result.html', query=query, metrics=metrics, top10_doc=top10_doc)
    return render_template('query.html')

if __name__ == '__main__':
    app.run()

