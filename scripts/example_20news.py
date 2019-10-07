

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn import metrics

# Bag of words  / ngrams, simple example:
vectorizer1 = CountVectorizer()

corpus = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']

X1 = vectorizer.fit_transform(corpus)

print(X1.toarray())  

print(vectorizer1.get_feature_names())

print(vectorizer1.transform(['This This second asdfasfd']).toarray())


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
 
X2=bigram_vectorizer.fit_transform(corpus)
print(bigram_vectorizer.get_feature_names())





#Bag of words on newsgroup:

newsgroups_train = fetch_20newsgroups(subset='train')
#compare with
#newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))


train_data=newsgroups_train.data 
Y_train = newsgroups_train.target

train_classes_names = newsgroups_train.target_names

nr_of_classes = len(train_classes_names)


vectorizer20 = CountVectorizer()
X_train = vectorizer20.fit_transform(train_data)

print(X_train.shape)

newsgroups_test = fetch_20newsgroups(subset='test')
#compare with
#newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


test_data = newsgroups_test.data 
Y_test=newsgroups_test.target

print("len(test_data) = ", len(X_data))

X_test  = vectorizer20.transform(test_data)

clf = MultinomialNB(alpha=.01)
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(Y_test,Y_pred))

 


