import numpy as np
import gensim
import argparse
import glob
import pickle

from nltk.corpus import brown, movie_reviews, treebank
from gensim.models import Word2Vec
from nltk import word_tokenize

# zakomentowac po pierwszym uruchomieniu! Uruchomic przed pierwszym uruchomieniem!!!

# nltk.download('punkt')
# nltk.download('brown')

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')

    args = parser.parse_args()

    return (args.input_dir, args.output_dir)


def read_data(path, classes_names):
    X = []
    Y = []
    for classs in classes_names:
        for file_name in list((glob.glob(path + classs + "/**"))):
            f = open(file_name, "r")
            text = f.read()
            X.append(text)
            Y.append(classes_dict[classs])
    return X, Y

input_dir, output_dir= ParseArguments()

classes_names = []

classes_dict = {}

counter = 0;

for file in glob.glob(input_dir + "/train/**"):
    tmp = file.rsplit('/', 3)
    classes_names.append(tmp[len(tmp) - 1])
    classes_dict[tmp[len(tmp) - 1]] = counter;
    counter = counter + 1;

# wczytujemy dane treningowe
X_train_data, Y_train = read_data(input_dir + "/train/", classes_names)

# zapisujemy nazwy klas do pliku
classes_names_path = output_dir + '/classes_names.pkl'

classes_names_outfile = open(classes_names_path, 'wb')

pickle.dump(classes_names, classes_names_outfile)

classes_names_outfile.close()

# trzeba sciagnac plik https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz
# Zapisac ten plik w folderze scripts
w2v = gensim.models.KeyedVectors.load_word2vec_format("scripts/GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)

nr_posts_train = len(X_train_data)

for i in np.arange(nr_posts_train):
    post = X_train_data[i] # bierzemy dany plik
    word_tokens = word_tokenize(post) # rozkladamy na pojedyncze slowka
    words = []
    for word in word_tokens:
        if word in w2v.vocab:
            words.append(word)
    words_mean = np.mean(w2v[words], axis=0).reshape(-1, 300) # w2v[words] wektor dla danego slowa i bierzemy srednia z kazdej kolumny
    if (i == 0):
        x_train = words_mean
    else:
        x_train = np.concatenate((x_train, words_mean))

x_train = np.c_[np.transpose(Y_train), x_train]

#zapisywanie do pliku
train_data_path = output_dir + '/train_data.pkl'

train_data_outfile = open(train_data_path, 'wb')

pickle.dump(x_train, train_data_outfile)

train_data_outfile.close()

# --------- dane testowe
X_test_data, Y_test = read_data(input_dir + "/test/", classes_names)
nr_posts_test = len(X_test_data)

for i in np.arange(nr_posts_test):
    post = X_test_data[i]
    word_tokens = word_tokenize(post)
    words = []
    for word in word_tokens:
        if word in w2v.vocab:
            words.append(word)
    words_mean = np.mean(w2v[words], axis=0).reshape(-1, 300)
    if (i == 0):
        x_test = words_mean
    else:
        x_test = np.concatenate((x_test, words_mean))

# zapisywanie do pliku
x_test = np.c_[np.transpose(Y_test), x_test]

test_data_path = output_dir + '/test_data.pkl'

test_data_outfile = open(test_data_path, 'wb')

pickle.dump(x_test, test_data_outfile)

test_data_outfile.close()


# ciekawostki dotyczace dzialania w2v
#print(words_mean)
#print(words_mean.size)
#print(w2v.similarity('computer','laptop'))
#print(w2v.most_similar('computer'))
