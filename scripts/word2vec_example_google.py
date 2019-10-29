import numpy as np
import time
import gensim
 
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
 
from gensim.models import Word2Vec
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#w2v = Word2Vec(brown.sents())

#trzeba sciagnac plik https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz
w2v = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)


from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', shuffle=True)
news_test = fetch_20newsgroups(subset='test', shuffle=True)



nr_posts_train = 2000
#nr_posts_train=len(news_train.data)

y_train=news_train.target[:nr_posts_train]


for i in np.arange(nr_posts_train):
	post=news_train.data[i]
	word_tokens = word_tokenize(post)
	words=[]
	for word in word_tokens:
		if word in w2v.vocab:
			words.append(word)
	words_mean  = np.mean(w2v[words],axis=0).reshape(-1,300)
	if(i==0):
		x_train=words_mean
	else:
		x_train = np.concatenate((x_train, words_mean))
	
	
# now we read in from test set, say 
nr_posts_test = 500
y_test=news_test.target[:nr_posts_test]


for i in np.arange(nr_posts_test):
	post=news_test.data[i]
	word_tokens = word_tokenize(post)
	words=[]
	for word in word_tokens:
		if word in w2v.vocab:
			words.append(word)
	words_mean  = np.mean(w2v[words],axis=0).reshape(-1,300)
	if(i==0):
		x_test=words_mean
	else:
		x_test = np.concatenate((x_test, words_mean))
			
	#print(words_mean)

print("Classification results:")
NB_clf = GaussianNB()
NB_clf.fit(x_train, y_train)
y_pred = NB_clf.predict(x_test)


print("Naive Bayes class. rate: \t", accuracy_score(y_pred,y_test))


knn=5
knn_clf = KNeighborsClassifier(n_neighbors=knn)
knn_clf.fit(x_train, y_train)
y_pred = knn_clf.predict(x_test)
print("kNN ( k=",knn,") class. rate: \t", accuracy_score(y_pred,y_test))
	
svm_clf = SVC(  gamma=15)
svm_clf.fit(x_train, y_train)
y_pred = svm_clf.predict(x_test)
print("SVM classification rate: \t", accuracy_score(y_pred,y_test))
	
	  
	
