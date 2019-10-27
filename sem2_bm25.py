#!/usr/bin/env python
# coding: utf-8

# ## Лекция 2  BM5    

# ## Функция ранжирования bm25

# Для обратного индекса есть общепринятая формула для ранжирования *Okapi best match 25* ([Okapi BM25](https://ru.wikipedia.org/wiki/Okapi_BM25)).    
# Пусть дан запрос $Q$, содержащий слова  $q_1, ... , q_n$, тогда функция BM25 даёт следующую оценку релевантности документа $D$ запросу $Q$:
# 
# $$ score(D, Q) = \sum_{i}^{n} \text{IDF}(q_i)*\frac{TF(q_i,D)*(k+1)}{TF(q_i,D)+k(1-b+b\frac{l(d)}{avgdl})} $$ 
# где   
# >$TF(q_i,D)$ - частота слова $q_i$ в документе $D$      
# $l(d)$ - длина документа (количество слов в нём)   
# *avgdl* — средняя длина документа в коллекции    
# $k$ и $b$ — свободные коэффициенты, обычно их выбирают как $k$=2.0 и $b$=0.75   
# $$$$
# $\text{IDF}(q_i)$ - это модернизированная версия IDF: 
# $$\text{IDF}(q_i) = \log\frac{N-n(q_i)+0.5}{n(q_i)+0.5},$$
# >> где $N$ - общее количество документов в коллекции   
# $n(q_i)$ — количество документов, содержащих $q_i$

# In[47]:


from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import csv
from math import log

morph = MorphAnalyzer()
vec = CountVectorizer()
vec_bin = CountVectorizer(binary=True)


# In[48]:


table_csv = pd.read_csv('quora_question_pairs_rus.csv', index_col='Unnamed: 0')


# In[49]:


def preproc(el):
    t = str(el)
    t = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"\ufeff]', r'', t)
    text = WordPunctTokenizer().tokenize(t)
    preproc_text = ''
    for w in text:
        new_w = morph.parse(w)[0].normal_form + ' '
        preproc_text += new_w
    return preproc_text

def query_mat(column): 
    texts_words = [] 
    idxs = [] 
    for idx, text in enumerate(column): 
        ws = preproc(text) 
        texts_words.append(ws) 
        idxs.append(idx) 
        if len(texts_words) % 1000 == 0: 
            print(f'{len(texts_words)} done') 
    global vec, vec_bin 
#     vec = CountVectorizer(min_df=5)
#     vec_bin = CountVectorizer(min_df=5, binary=True) 
    X = vec.fit_transform(texts_words) 
    Y = vec_bin.fit_transform(texts_words) 
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=idxs) 
    df_bin = pd.DataFrame(Y.toarray(), columns=vec.get_feature_names(), index=idxs) 
    return df, df_bin 

def doc_vectorizer(column): 
    texts_words = [] 
    idxs = [] 
    for idx, text in enumerate(column): 
        ws = preproc(text) 
        texts_words.append(ws) 
        idxs.append(idx) 
        if len(texts_words) % 1000 == 0: 
            print(f'{len(texts_words)} done') 
    X = vec.transform(texts_words) 
    Y = vec_bin.transform(texts_words) 
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=idxs) 
    df_bin = pd.DataFrame(Y.toarray(), columns=vec.get_feature_names(), index=idxs) 
    return df, df_bin


# In[50]:


def create_inverse_counter(binary_1, binary_q):
    word_counter_dict = {}
    for column in binary_q.columns:
        col = binary_1[column]
        summa = col.sum()
        word_counter_dict[column] = summa
    inverse_counter = pd.DataFrame.from_dict(word_counter_dict, orient='index')
    inverse_counter = inverse_counter.transpose()
    return inverse_counter

def create_idf_table(ids, N): 
    idfs = {}
    for w in ids:
        idf = log((N - ids[w] + 0.5)/(ids[w] +0.5))
        idfs[w] = idf
    idf_table = pd.DataFrame.from_dict(idfs, orient='index')
    idf_table = idf_table.transpose()
    return idf_table  

def create_tf_table(df):
    df['sum'] = df.sum(axis=1)
    tf_table = df.div(df['sum'], axis=0)
    tf_table = tf_table.fillna(0)
    return tf_table

def dl_avgdl(data):
    data['sum'] = data.sum(axis = 1, skipna = True) 
    sums = data['sum']
    avg = data['sum'].mean()
    sums_normalized = sums.div(avg)
    return sums_normalized, avg  


# In[51]:


# little_corp = corpus_query[:10000] #значит у нас есть повторяющиеся query
# print(len(set(little_corp)))


# In[52]:


def dict_relev(data):
    num = 0
    id_relev={}
    set_query ={}
    with open (data, 'r') as file:
        csv_file = csv.reader(file, delimiter =',')
        for row in csv_file:
            num+=1
            id_row = row[0]
            query = row[1]
            res_relev = row[3]
            if res_relev == "1":
                if query in set_query:
                    id_q=set_query[query]
                    id_relev[id_q].append(id_row)
                else:
                    set_query[query] = id_row
                    id_relev[id_row]=[id_row]
            else:
                pass
    return id_relev


# In[53]:


q_df, q_df_bin = query_mat(table_csv['question1'][:50])
doc_df, doc_df_bin = doc_vectorizer(table_csv['question2'][:50])

q_df = q_df.fillna(0) #заполняем нули
q_df_bin = q_df_bin.fillna(0)
doc_df = doc_df.fillna(0)
doc_df_bin = doc_df_bin.fillna(0)

is_dupl = table_csv['is_duplicate'][:50]

dls, avg_dl = dl_avgdl(doc_df) 

relevant_data = dict_relev('quora_question_pairs_rus.csv')

tf_table = create_tf_table(q_df)

inv_df = create_inverse_counter(q_df_bin, doc_df_bin)

idfs = create_idf_table(inv_df, q_df_bin.shape[0])


# In[81]:


str(4) in relevant_data[str(4)]


# In[54]:


q_df


# In[55]:


queries = table_csv['question1'][:50]


# In[56]:


# for query in queries:
#     print (query)


# In[57]:


k = 2.0
b = 0.75

def bm_25(doc_idx, query, wordlist, docs_num, tfs, idfs, c1, c2):
    bm_val = 0
    for w in wordlist[:-1]:
        if query[w] != 0:
            idf = float(idfs[w])
            tf_value = float(tfs.iloc[doc_idx][w])
            bm_i = idf * ((tf_value * c1)/(tf_value + c2))
            bm_val += bm_i
    return bm_val


# In[ ]:


bm_25()


# In[103]:


def final(doc_data, q_data, relevant_data, is_rel, tfs, idfs, dls_norm, k, b):
    res_table = pd.DataFrame(columns=['query', 'relevant_docs', 'res'])
    N = doc_data.shape[0] #кол-во доков
    docs = []
    lemmas_list = list(q_data.columns) #список слов (один)
    const_high = k + 1 #верхняя, неизменяемая
    for q_id, q_words in q_data.iterrows():
        relevance_dict = {}
        for doc_id, doc_words in doc_data.iterrows():
            len_norm = dls_norm[doc_id] #длина док/на сред длин док
            const_low = k * (1 - b + b * len_norm)#len_doc/avgdl нижняя, изменяемая
            bm_25 = func_bm_25(doc_id, q_words, lemmas_list, N, tfs, idfs, const_high, const_low)
            relevance_dict[bm_25] = doc_id #бм25 - номер дока
            docs.append(doc_id)
        docs_sorted = sorted(relevance_dict.items(), reverse=True)
        best_5_docs = [el[1] for el in docs_sorted[:5]]
        points = 0
        for doc in docs:
            if str(q_id) in relevant_data:
                if str(doc) in relevant_data[str(q_id)] and doc in best_5_docs:
                    points +=1
#         if str(q_id) in relevant_data:
#             print (q_id)
#             print (relevant_data[str(q_id)])
#             print (doc_id)                   
#             if doc_id in relevant_data[str(q_id)]:
#                 print ('yessss 1') #and doc_id in best_5_docs:
#                 points += 1
        if points > 0:
            points = 1
        res_table = res_table.append({'query': q_id, 'relevant_docs': best_5_docs, 'res':points}, ignore_index=True)
        rel_sorted = {} #обнуляем
        best_5_rel = []
    return res_table        


# In[104]:


results = final (doc_df, q_df, relevant_data, is_dupl, tf_table, idfs, dls, k, b)


# In[105]:


results


# In[107]:


from sklearn.metrics import accuracy_score
#функция метрики (кирилллл подсказал)
def metric(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = accuracy_score(y_true, y_pred)
    return acc

metric(is_dupl, results['res'])


# # Матрицы ура

# In[ ]:





# In[ ]:


q_df, q_df_bin = query_mat(table_csv['question1'][:10000])
doc_df, doc_df_bin = doc_vectorizer(table_csv['question2'][:10000])

q_df = q_df.fillna(0) #заполняем нули
q_df_bin = q_df_bin.fillna(0)
doc_df = doc_df.fillna(0)
doc_df_bin = doc_df_bin.fillna(0)

is_dupl = table_csv['is_duplicate'][:10000]

dls, avg_dl = dl_avgdl(doc_df) 

relevant_data = dict_relev('quora_question_pairs_rus.csv')

tf_table = create_tf_table(q_df)

inv_df = create_inverse_counter(q_df_bin, doc_df_bin)

idfs = create_idf_table(inv_df, q_df_bin.shape[0])

def matrix_multiplication(queries, docs, tfs, idfs, avg_lens, k, b):
    tfs = tfs.drop(columns=['sum'])
    conversion_table = queries.mul(tfs)
    conversion_table_numerator = conversion_table.mul(k+1)
    coefficient = avg_lens.mul(b)
    coefficient = coefficient.add(1-b)
    coefficient = coefficient.mul(k)
    conversion_table_denominator = conversion_table.mul(avg_lens, axis=0)
    tf_factor = conversion_table_numerator.divide(conversion_table_denominator)
    tf_factor = tf_factor.fillna(0)
    n = tf_factor.shape[0]
    idf_table = pd.concat([idfs]*n, ignore_index=True) 
    term_table = tf_factor.mul(idf_table, axis=1)
    docs = docs.T
    bm_25_df = term_table.dot(docs)
    return bm_25_df

def matrix_best_5(d, correspondence, is_rel):
    best_5_df = pd.DataFrame(columns=['idx_q1', 'relevant_docs', 'match'])
    d = d.fillna(0)
    for q_idx, q_row in d.iteritems():
        q_row = q_row.astype('int64')
        best_5 = list(q_row.nlargest(5).index)
        matches = 0
        for d_idx in correspondence[q_idx]:
            if d_idx in best_5 and is_rel[d_idx] == 1:
                matches += 1
        if matches > 0:
            matches = 1
        best_5_df = best_5_df.append({'idx_q1': q_idx, 'relevant_docs': best_5, 'match':matches}, ignore_index=True)
    return best_5_df

bm_25_matrices = matrix_multiplication(q1_df_bin, q2_df_bin, tf_table, idfs, dls, k, b)

results_matrix = matrix_best_5(bm_25_matrices, relevant_data, is_dupl)


# ### __Задача 1__:    
# Напишите два поисковика на *BM25*. Один через подсчет метрики по формуле для каждой пары слово-документ, второй через умножение матрицы на вектор. 
# 
# Сравните время работы поиска на 100к запросах. В качестве корпуса возьмем 
# [Quora question pairs](https://www.kaggle.com/loopdigga/quora-question-pairs-russian).

# In[ ]:





# ### __Задача 2__:    
# 
# 

# Выведите 10 первых результатов и их близость по метрике BM25 по запросу **рождественские каникулы** на нашем корпусе  Quora question pairs. 

# ### __Задача 3__:    
# 
# Посчитайте точность поиска при 
# 1. BM25, b=0.75 
# 2. BM15, b=0 
# 3. BM11, b=1
