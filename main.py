import proc_func
import pandas as pd

import nltk
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB


import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

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


english_words = set(nltk.corpus.words.words())
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
count_vect = CountVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,3), stop_words = stop_words)

# Transform the txt from the training set and test set into the matrix representation of the frequency of their tokens
Txt_train = count_vect.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test = count_vect.transform(txt_test)





# When setting a parameter to loop through the form is: function name + __ + parameter
n_parameters  = {'countvectorizer__ngram_range': [(1, 1),(1, 2), (1, 3)]}

# Linear regression output scores vs ngram method
# The pipeline allows for one function to be preformed before another and the GridSearch 
# iterates through the parameters

count_vect_1 = CountVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,1), stop_words = stop_words)
Txt_train_1 = count_vect_1.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_1 = count_vect_1.transform(txt_test)

count_vect_2 = CountVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,2), stop_words = stop_words)
Txt_train_2 = count_vect_2.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_2 = count_vect_2.transform(txt_test)

count_vect_3 = CountVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,3), stop_words = stop_words)
Txt_train_3 = count_vect_3.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_3 = count_vect_3.transform(txt_test)

gnb_1 = GaussianNB()
gnb_1.fit(Txt_train_1.toarray(), cat_train)
gnb_2 = GaussianNB()
gnb_2.fit(Txt_train_2.toarray(), cat_train)
gnb_3 = GaussianNB()
gnb_3.fit(Txt_train_3.toarray(), cat_train)

lr_1 = LogisticRegressionCV()
lr_1.fit(Txt_train_1.toarray(), cat_train)
lr_2 = LogisticRegressionCV()
lr_2.fit(Txt_train_2.toarray(), cat_train)
lr_3 = LogisticRegressionCV()
lr_3.fit(Txt_train_3.toarray(), cat_train)


# The TF-IDF vectorizer is applied then the LogisticRegression algorithm is applied

count_vect_1_tf = TfidfVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,1), stop_words = stop_words)
Txt_train_1_tf = count_vect_1_tf.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_1_tf = count_vect_1_tf.transform(txt_test)

count_vect_2_tf = TfidfVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,2), stop_words = stop_words)
Txt_train_2_tf = count_vect_2_tf.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_2_tf = count_vect_2_tf.transform(txt_test)

count_vect_3_tf = TfidfVectorizer(tokenizer = LemmaTokenizer, min_df=30, ngram_range = (1,3), stop_words = stop_words)
Txt_train_3_tf = count_vect_3_tf.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_3_tf = count_vect_3_tf.transform(txt_test)

gnb_1_tf = GaussianNB()
gnb_1_tf.fit(Txt_train_1_tf.toarray(), cat_train)
gnb_2_tf = GaussianNB()
gnb_2_tf.fit(Txt_train_2_tf.toarray(), cat_train)
gnb_3_tf = GaussianNB()
gnb_3_tf.fit(Txt_train_3_tf.toarray(), cat_train)

lr_1_tf = LogisticRegressionCV()
lr_1_tf.fit(Txt_train_1_tf.toarray(), cat_train)
lr_2_tf = LogisticRegressionCV()
lr_2_tf.fit(Txt_train_2_tf.toarray(), cat_train)
lr_3_tf = LogisticRegressionCV()
lr_3_tf.fit(Txt_train_3_tf.toarray(), cat_train)

#Calculate the accuracy of each method
gnb_1_s = gnb_1.score(Txt_test_1.toarray(), cat_test)
gnb_2_s = gnb_2.score(Txt_test_2.toarray(), cat_test)
gnb_3_s = gnb_3.score(Txt_test_3.toarray(), cat_test)
gnb_1_tf_s = gnb_1_tf.score(Txt_test_1_tf.toarray(), cat_test)
gnb_2_tf_s = gnb_2_tf.score(Txt_test_2_tf.toarray(), cat_test)
gnb_3_tf_s = gnb_3_tf.score(Txt_test_3_tf.toarray(), cat_test)
lr_1_s = lr_1.score(Txt_test_1.toarray(), cat_test)
lr_2_s = lr_2.score(Txt_test_2.toarray(), cat_test)
lr_3_s = lr_3.score(Txt_test_3.toarray(), cat_test)
lr_1_tf_s = lr_1_tf.score(Txt_test_1_tf.toarray(), cat_test)
lr_2_tf_s = lr_2_tf.score(Txt_test_2_tf.toarray(), cat_test)
lr_3_tf_s = lr_3_tf.score(Txt_test_3_tf.toarray(), cat_test)


gnb_scores = [gnb_1_s, gnb_2_s, gnb_3_s]
lr_scores = [lr_1_s, lr_2_s, lr_3_s]
lr_tf_scores = [lr_1_tf_s, lr_2_tf_s, lr_3_tf_s]
gnb_tf_scores = [gnb_1_tf_s, gnb_2_tf_s, gnb_3_tf_s]
print('Accuracy as a function of n-gram range')
print('Gaussian Naive Bayes Accuracy:', gnb_scores)
print('Gaussian Naive Bayes TF-IDF Weighted Accuracy:',gnb_tf_scores)
print('Logistic Regression Accuracy:', lr_scores)
print('Logistic Regression TF-IDF Weighted Accuracy:',lr_tf_scores)


algorithm_typ = ['Gaussian Naive Bayes','Gaussian TF-IDF Naive Bayes' 'Logistic', 'TF-IDF Logistic',]
n = ['N=1', 'N=2','N=3']

fig = go.Figure()
fig.add_trace(go.Bar(x = n, y = gnb_scores, name = 'Gaussian Naive Bayes', marker_color = 'lightblue', width = 0.2, text = gnb_scores))
fig.add_trace(go.Bar(x = n, y = gnb_tf_scores, name = 'TF-IDF Gaussian Naive Bayes', marker_color = 'blue', width = 0.2, text = gnb_tf_scores))

fig.add_trace(go.Bar(x = n, y = lr_scores, name = 'Logistic Regression', marker_color = 'plum', width = 0.2, text = lr_scores))
fig.add_trace(go.Bar(x = n, y = lr_tf_scores, name = 'TF-IDF Logistic Regression', marker_color = 'purple', width = 0.2, text = lr_tf_scores))
fig.update_traces(texttemplate = '%{text:.3f}', textposition = 'outside', textfont_size = 12)
fig.update_layout(title='Accuracy of Machine Learning Algorithms vs N-Gram Value',
                  uniformtext_minsize = 12,
    xaxis_tickfont_size = 12, yaxis = dict(title = 'Accuracy (%)', titlefont_size = 13, tickfont_size = 12),
    legend = dict(x = 1, y = 1), barmode = 'group', bargap = .1, bargroupgap = .3)

fig.write_image('fig1.png',scale=5)

# Look at the effect of the C value on the Logistic Regression Accuracy
parameters  = {"logisticregression__C": [0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1),(1, 2), (1, 3)]} #, (1, 2), (1, 3)

# The pipeline allows for one function to be preformed before another
# The TF-IDF vectorizer is applied then the LogisticRegression algorithm is applied
pipeline = make_pipeline(TfidfVectorizer(tokenizer = LemmaTokenizer, min_df=30, stop_words = stop_words), LogisticRegression())

grid_search = GridSearchCV(pipeline, parameters)
grid_search.fit(txt_train, cat_train)

# extract scores from grid_search
scores = grid_search.cv_results_['mean_test_score'].reshape(-1, 3).T
print(scores)
ordered_scores = [scores[0],scores[2],scores[1]]
print(ordered_scores)
s_text = np.around(ordered_scores, decimals=4) # Only show rounded value (full value on hover)


# visualize heat map
fig0 = ff.create_annotated_heatmap(s_text, showscale=True,
               y = ['n-grams = (1, 1)','n-grams = (1, 2)', 'n-grams = (1, 3)'], 
                                  x = ['C = 0.01','C = 0.1','C = 1','C = 10','C = 100'])

fig0.write_image('heatmap.png',scale=5)


# Return what the sklearn toolkit found as the best value of the C parameter for the (1,3)
# n-gram model
print('Most effective C value: ',  lr_3_tf.C_[0])
c_eff = lr_3_tf.C_[0]
lr_3_tf_eff = LogisticRegression(C=c_eff)
lr_3_tf_eff.fit(Txt_train_3_tf.toarray(), cat_train)
print(lr_3_tf_eff.score(Txt_test_3_tf.toarray(), cat_test))

feature_names = count_vect_3_tf.get_feature_names()
term_imp = lr_3_tf_eff.coef_.ravel() #Gives the term importance after ifidf weighting
tech_imp = np.argsort(term_imp)[-20:]
tech_imp = [int(x) for x in tech_imp]
feature_names_tech = [feature_names[x] for x in tech_imp]
nontech_imp = np.argsort(term_imp)[:20]
feature_names_nontech = [feature_names[x] for x in nontech_imp]

tot_imp = np.concatenate([nontech_imp,tech_imp])
tot_features = np.concatenate([feature_names_nontech,feature_names_tech])


fig2 = go.Figure()

fig2.add_trace(go.Bar(x = tot_features,
                     y = term_imp[tot_imp], 
                      width = 0.4, text = gnb_scores))
fig2.update_layout(title='Coefficient Value for 40 Most Influential Terms',
                      xaxis_tickfont_size = 10,uniformtext_mode='show')
fig2.write_image('fig3.png',scale=5)

def LemmaTokenizer_altered(text):
    tokens = reg.tokenize(text) # tokenize the text
    lemmas = list()
    for token in tokens:
        if token.isalpha() and token not in stop_words and len(token)>1:
            lemma = lemmatizer.lemmatize(token)
            lemmas.append(lemma)
        if token == 'References':
            break
    return lemmas
count_vect_3_tf_alt = TfidfVectorizer(tokenizer = LemmaTokenizer_altered, min_df=30, ngram_range = (1,3), stop_words = stop_words)
Txt_train_3_tf_alt = count_vect_3_tf_alt.fit_transform(txt_train) # Also preform a fit on the txt_train which creates a dictionary of all the tokens in the documents
Txt_test_3_tf_alt = count_vect_3_tf_alt.transform(txt_test)
lr_3_tf_alt_eff = LogisticRegression(C=c_eff)
lr_3_tf_alt_eff.fit(Txt_train_3_tf_alt.toarray(), cat_train)
print('After removing tokens with length one the accuracy is:',
      lr_3_tf_alt_eff.score(Txt_test_3_tf_alt.toarray(), cat_test))
