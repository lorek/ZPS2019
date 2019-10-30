import numpy as np
import sys
import argparse

import glob, os

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn import metrics


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--data-dir', default="", required=True, help='data dir')

    args = parser.parse_args()

    return args.data_dir


def read_data(path, classes_names):
    print("classes_names = ", classes_names)
    X = []
    Y = []
    for classs in classes_names:
        # search for all files in path
        for file_name in list((glob.glob(path + classs + "/**"))):
            print("klasa = ", classs, ", file = ", file_name)
            f = open(file_name, "r")
            text = f.read()
            X.append(text)
            Y.append(classes_dict[classs])

    return X, Y


# main progam
data_dir = ParseArguments()

print("data-dir = ", data_dir)

# folder data_dir/train should contain only subfolders = classes



# first, we read only subfolders names in   data_dir/train
classes_names = []

classes_dict = {}

counter = 0;

for file in glob.glob(data_dir + "/train/**"):
    tmp = file.rsplit('/', 3)
    classes_names.append(tmp[len(tmp) - 1])
    classes_dict[tmp[len(tmp) - 1]] = counter;
    counter = counter + 1;

print("Klasy  = ", classes_names)

# read in train data
X_train_data, Y_train = read_data(data_dir + "/train/", classes_names)

# BoW
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_data)

# classifier
clf = MultinomialNB(alpha=.01)
clf.fit(X_train, Y_train)

# read in test data
X_test_data, Y_test = read_data(data_dir + "/test/", classes_names)
X_test = vectorizer.transform(X_test_data)

Y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))

