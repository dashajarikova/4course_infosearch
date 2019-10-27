import os
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import nltk
import collections
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
X = vec.fit_transform(corpus)

df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=names_docs)

papka = 'friends'
print(os.path.abspath('Friends - 2x01 - The One With Ross\'s New Girlfriend.ru.txt'))


corpus = []
ep_per_seasons = {}
names_docs = []

main_dir = '/friends'
folders = [f for f in os.listdir(main_dir) if 'Store' not in f]


def preproc(el):
    t = str(el)
    t = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"\ufeff]', r'', t)
    text = WordPunctTokenizer().tokenize(t)
    preproc_text = ''
    for w in text:
        new_w = morph.parse(w)[0].normal_form + ' '
        preproc_text += new_w
    return preproc_text

def search(query):
    query_preproc = preprocessing(query)
    vec_query = vec.transform([query_preproc])
    vec_query = vec.transform([query_preproc])
    results = {}
    for index, row in df.iterrows():
        vector = row.as_matrix()
        cos_sim = cosine_similarity(vector.reshape(1, -1), vec_query)
        cos_sim = np.asscalar(cos_sim)
        results[cos_sim] = index
    return results
    
for folder in folders:
    for file in os.listdir(os.path.join(main_dir, folder)):
        filepath = os.path.join(main_dir, folder, file)
        names_docs.append(file)
        with open(filepath, 'r') as f:
            text = f.read()
        preproc_text = preproc(text)
        corpus.append(preproc_text)
        ep_per_seasons[file] = folder

query = input('Введите запрос: ')
results = search(query)
print('10 самых подходящих серий: ')
for i, key in enumerate(sorted(results, reverse=True)[:10]):
    print(str(i+1) + ': ' + results[key]) 

#a) какое слово является самым частотным
#b) какое самым редким
dic = {}
for c in vec.get_feature_names():
    dic[sum(df[c])] = c
print('Самое частотное слово:', dic[max(dic.keys())], max(dic.keys()))
print('Самое редкое слово:', dic[min(dic.keys())], min(dic.keys()))

#c) какой набор слов есть во всех документах коллекции
in_all_docs = []
d = df.isin([0])
for c in vec.get_feature_names():
    if sum(d[c]) == 0:
        in_all_docs.append(c)
print('Слова, которые есть во всех текстах:', ', '.join(in_all_docs))

#e) кто из главных героев статистически самый популярный?
d = {}
characters = ['росс', 'фиби', 'моника', 'чендлер', 'джо', 'рэйчел']
for character in characters:
    try:
        d[sum(df[character])] = character
    except KeyError:
        break
print('Самое популярный герой:', d[max(d.keys())], max(d.keys()))


# In[ ]:




