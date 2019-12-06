
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import glob
import numpy as np
import pickle


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

# read in train data
X_train_data, Y_train = read_data(input_dir + "/train/", classes_names)


vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_data)
X_train = X_train.toarray()
Y_train = np.array(Y_train)


tfidf_data_train = np.c_[np.transpose(Y_train), X_train]

# zapisywanie do pliku

train_data_path = output_dir + '/train_data.pkl'
train_data_outfile = open(train_data_path, 'wb')
pickle.dump(tfidf_data_train, train_data_outfile)
train_data_outfile.close()

train_data_infile = open(train_data_path, 'rb')
print(pickle.load(train_data_infile))
train_data_infile.close()

X_test_data, Y_test = read_data(input_dir + "/test/", classes_names)
X_test = vectorizer.transform(X_test_data)
X_test = X_test.toarray()
Y_test = np.array(Y_test)



tfidf_data_test = np.c_[np.transpose(Y_test), X_test]

# zapisywanie do pliku
# test_data_path = output_dir + '/test_data.pkl'
# test_data_outfile = open(test_data_path, 'wb')
# pickle.dump(tfidf_data_test, test_data_outfile)
# test_data_outfile.close()


