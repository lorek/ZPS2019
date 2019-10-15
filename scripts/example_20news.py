

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn import metrics

# Bag of words  / ngrams, simple example:

print("EXAMPLE 1:")

vectorizer1 = CountVectorizer()

corpus = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']

print("corpus = ", corpus)

X1 = vectorizer1.fit_transform(corpus)

print("1 GRAM")
print("'Wydobyte' slowa/cechy:", vectorizer1.get_feature_names())

print("Macierz: (zdanie x slowa):", X1.toarray())  
 
 
nowe_zdanie= 'This This second asdfasfd'
print("Zapsisanie zdania '", nowe_zdanie,"' w powyzszych cechach")
print(vectorizer1.transform([nowe_zdanie]).toarray())



print("2 GRAM")
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
 
X2=bigram_vectorizer.fit_transform(corpus)

print("Cechy: ",bigram_vectorizer.get_feature_names())



#Bag of words on newsgroup:

print("EXAMPLE 2 (20 newsgroups)")
newsgroups_train = fetch_20newsgroups(subset='train')
#compare with
#newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))


train_data=newsgroups_train.data 
Y_train = newsgroups_train.target

train_classes_names = newsgroups_train.target_names

nr_of_classes = len(train_classes_names)


vectorizer20 = CountVectorizer()
X_train = vectorizer20.fit_transform(train_data)

print("Dane treningowe, rozmiar: ", X_train.shape)

newsgroups_test = fetch_20newsgroups(subset='test')
#compare with
#newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


test_data = newsgroups_test.data 
Y_test=newsgroups_test.target

print("len(test_data) = ", len(test_data))

X_test  = vectorizer20.transform(test_data)

clf = MultinomialNB(alpha=.01)

print("Uczymy klasyfikator (multinomial) naive Bayes...")
clf.fit(X_train,Y_train)

print("Co/jak klasyfikator 'przewiduje' na danych testowych:")
Y_pred=clf.predict(X_test)

print("Wyink - dokladnosc (accuracy): ", metrics.accuracy_score(Y_test,Y_pred))

 


