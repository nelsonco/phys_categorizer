# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:37:00 2020

@author: Court
"""

import proc_func
import os 
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import nltk.tokenize as tok
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words') 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import mglearn
import matplotlib as plt


main_directory = r'C:\Users\Court\Downloads\physics'

# Select categories that will fall into binary categories for technical and non-technical
non_technical_folders = ['physics.pop-ph','physics.ed-ph','physics.gen-ph','physics.hist-ph',
                         'physics.soc-ph']
technical_folders = ['physics.acc-ph','physics.ao-ph', 'physics.atm-clus','physics.atom-ph',
                    'physics.bio-ph','physics.chem-ph','physics.class-ph','physics.comp-ph',
                    'physics.data-an','physics.flu-dyn','physics.geo-ph','physics.ins-det',
                    'physics.med-ph', 'physics.optics','physics.plasm-ph','physics.acc-ph']

# Create a DataFrames of the Technical and Non-Technical Categories
count = 0
for folder in non_technical_folders:
    spec_dir = main_directory + '\\'+folder
    df = proc_func.NewDataFrame(spec_dir, 'files')
    if count == 0:
        df_nontech = df
    else:
        df_nontech = pd.concat([df_nontech, df])
    count = count + 1

count = 0      
for folder in technical_folders:
    spec_dir = main_directory + '\\'+folder
    df = proc_func.NewDataFrame(spec_dir, 'files')
    if count == 0:
        df_tech = df
    else:
        df_tech = pd.concat([df_tech, df])
    count = count + 1
    
# Relabel the category of each document as Technical or Non-Technical

df_nontech = df_nontech.drop(columns=['Category'])
df_nontech.insert(1, "Category", 'non-technical')
df_tech = df_tech.drop(columns=['Category'])
df_tech.insert(1, "Category", 'technical')

# Ensure there is an equal number of data points in the non-technical and technical categories
# This assumes that the technical category origionally has more data points

df_tech_sub = df_tech.sample(n=len(df_nontech)) # get a random sampling of the dataframe

# Merge the df_nontech and df_tech_sub dataframes
df_total = pd.concat([df_nontech, df_tech_sub])

# System for Tokenizing Text
reg = tok.RegexpTokenizer(r'\w+') #Tokenizes without punctuation
lemmatizer = WordNetLemmatizer()  #


stop_words = text.ENGLISH_STOP_WORDS.union(['phys', 'rev', 'pop', 'mi', 'al'])


def LemmaTokenizer(text):
    tokens = reg.tokenize(text) # tokenize the text
    lemmas = list()
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            lemmas.append(lemma)
        if token == 'References':
            break
    return lemmas
# Compile a list of the text values and corresponing categories from the dataframe
txts = list()
categories = list()
for index, row in df_total.iterrows():
    txt = row['Text']
    cat = row['Category']
    txts.append(txt)
    categories.append(cat)
# Separate the data randomly into training and testing sets, the defult of the train_test_splits
# is a random selection using np.random.Setting the test_size or train_size determines what precentage of the
# data is put in the test or training data sets. For example, 0.25 repressents 25%
txt_train, txt_test, cat_train, cat_test = train_test_split(txts, categories, stratify=categories,
                                                            test_size = 0.25)

# The CountVerctorizer() takes the text tokenizes it with the LemmaTokenizer defined above,
# Removes any tokens that do not occur in a minimum of three documents and in this case the
# n-grams stored are those less that or equal to three. Additionally stop words which has been
# defined above are removed. The count vectorizer stores the text as vectors which represent the 
# number of occurances of a given token within a document.
count_vect = CountVectorizer(tokenizer = LemmaTokenizer, min_df=3, ngram_range = (1,3), stop_words = stop_words)

# Transform the txt from the training set and test set into the matrix representation of the frequency of their tokens
Txt_train = count_vect.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test = count_vect.transform(txt_test)


feature_names = count_vect.get_feature_names()

print(feature_names[:10])

parameters  = {"logisticregression__C": [0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1),(1, 2), (1, 3)]} #, (1, 2), (1, 3)

# The pipeline allows for one function to be preformed before another
# The TF-IDF vectorizer is applied then the LogisticRegression algorithm is applied
pipeline = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())

print('done')
grid_search = GridSearchCV(pipeline, parameters)
print('done')
grid_search.fit(txt_train, cat_train)

import plotly.express as px

# extract scores from grid_search
scores = grid_search.cv_results_['mean_test_score'].reshape(-1, 3)


# visualize heat map
fig = px.imshow(scores, labels = dict(x = 'N-Gram Value', y = 'C Value', color = 'Accuracy(%)'),
                x = ['(1, 1)','(1, 2)', '(1, 3)'], y = ['C=100','C=10'])
fig.show()

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer

# When setting a parameter to loop through the form is: function name + __ + parameter
n_parameters  = {'countvectorizer__ngram_range': [(1, 1),(1, 2), (1, 3)]}

# Linear regression output scores vs ngram method
# The pipeline allows for one function to be preformed before another and the GridSearch 
# iterates through the parameters

pipeline_lr = make_pipeline(CountVectorizer(min_df=100), LogisticRegressionCV())
grid_search_lr = GridSearchCV(pipeline_lr, n_parameters)
grid_search_lr.fit(txt_train, cat_train)

pipeline_gnb = make_pipeline(CountVectorizer(min_df=100), GaussianNB())
grid_search_gnb = GridSearchCV(pipeline_gnb, n_parameters)
grid_search_gnb.fit(txt_train, cat_train)


# The TF-IDF vectorizer is applied then the LogisticRegression algorithm is applied
pipeline_lr_tfidf = make_pipeline(TfidfVectorizer(min_df=100), LogisticRegressionCV())
grid_search_lr_tfidf = GridSearchCV(pipeline, parameters)
grid_search_lr_tfidf.fit(txt_train, cat_train)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


gnb_p_3 = [gnb_score, gnb_score]
lr_p_3 = [lr_score, lr_score]
tfidf_p_3 = [tfidf_lr_score, tfidf_lr_score]

colors = ['lavenderblush', 'lawngreen', 'lightblue']

algorithm_typ = ['Naive', 'Logistic', 'TF-IDF Logistic',]
n = ['N=1', 'N=2']

fig = go.Figure()
fig.add_trace(go.Bar(x = n, y = gnb_p, name = 'Gausian Naive Bayes', marker_color = 'blue', width = 0.1, text = gnb_p))
fig.add_trace(go.Bar(x = n, y = lr_p, name='Logistic Regression', marker_color = 'green', width = 0.1, text = lr_p))
fig.add_trace(go.Bar(x = n, y = tfidf_p, name='TF-IDF Logistic Regression', marker_color = 'purple', width = 0.1, text = tfidf_p))
fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_layout(title='Accuracy of Machine Learning Algorithms vs N-Gram Value',
    xaxis_tickfont_size=14, yaxis=dict(title='Acuracy (%)', titlefont_size=16, tickfont_size=14,),
    legend = dict(x = 1, y = 1), barmode='group', bargap=.6, bargroupgap=.3)

fig.show()
