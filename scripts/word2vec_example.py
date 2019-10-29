import numpy as np
import time
 
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
import gensim
 

print("Calculating Word2Vec for 'brown' dataset .", end="", flush=True)
start_time = time.time()
w2v_brown = Word2Vec(brown.sents())
print("\t\t took %s seconds " % round((time.time() - start_time),5))
	
		
		 
print("Calculating Word2Vec for 'movie_reviews' dataset .", end="", flush=True)
start_time = time.time()
w2v_mr = Word2Vec(movie_reviews.sents())
print("\t\t took %s seconds " % round((time.time() - start_time),5))
	
		
		 
print("Calculating Word2Vec for 'treebank' dataset .", end="", flush=True)
start_time = time.time()
w2v_tb = Word2Vec(treebank.sents())
print("\t\t took %s seconds " % round((time.time() - start_time),5))

print("Reading in word2vec model GoogleNews_SLIM: .", end="", flush=True)
start_time = time.time()
w2v = gensim.models.KeyedVectors.load_word2vec_format("../data/text_word2vec_slim/GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)
print("\t\t took %s seconds " % round((time.time() - start_time),5))

print("\n") 

# ~ print("Google: most similar in 'brown': ",w2v_brown.most_similar("Google"))
# ~ print("Google: most similar in 'movie review': ",w2v_mr.most_similar("Google"))
# ~ print("Google: most similar in 'tree bank': ",w2v_tb.most_similar("Google"))

print("Google: most similar in 'GoogleNewsSLIM': ",w2v.most_similar("Google"))
print("\n");

print("house: most similar in 'GoogleNewsSLIM': ",w2v.most_similar("house"))
print("\n");

print("house: most similar in 'brown': ",w2v_brown.most_similar("house"))
print("\n");

print("house: most similar in 'movie review': ",w2v_mr.most_similar("house"))
print("\n");

print("house: most similar in 'tree bank': ",w2v_tb.most_similar("house"))
print("\n");




print("Germany - Berlin + Parisw in 'brown': ",w2v_brown.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1))
print("\n");
print("Germany - Berlin + Parisw in 'GoogleNewsSLIM': ",w2v.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1))
print("\n");

#print("Germany - Berlin + Parisw in 'movie review': ",w2v_mr.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1))
#print("Germany - Berlin + Parisw in 'tree bank': ",w2v_tb.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1))

