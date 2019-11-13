from sklearn.feature_extraction.text import CountVectorizer
import argparse
import glob
import numpy as np
import pickle

from sklearn.naive_bayes import MultinomialNB

# Przekazanie sciezek do plikow wejsciowych i wyjsciowych.
# Wejsciowe, np: --input-dir datasets/sample_dataset/
# Wyjsciowe, np: --output-dir zbior_A/sample_dataset_bow_2123

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')

    args = parser.parse_args()

    return (args.input_dir, args.output_dir)

# Odczytanie sciezek do plikow wejsciowych
def read_data(path, classes_names):
    # print("classes_names = ", classes_names)
    X = []
    Y = []
    for classs in classes_names:
        # search for all files in path
        for file_name in list((glob.glob(path + classs + "/**"))):
            #print("klasa = ", classs, ", file = ",file_name )
            #13.11.2019: Zmienilem "r" na "rb" i dodalem ponizsze dekodowanie/kodowanie
            f = open(file_name, "rb")
            text = f.read()
            text=text.decode('iso-8859-1').encode('utf8')
            X.append(text)
            Y.append(classes_dict[classs])
    return X, Y


# X - teksty z calego folderu w jednej zmiennej
# Y - klasy 0(computers),1(politics),2(sports)


# main progam - tu sie zaczyna

input_dir, output_dir= ParseArguments()
# print("input-dir = ", input_dir)
# print("output-dir = ", output_dir)

# folder data_dir/train should contain only subfolders = classes
# first, we read only subfolders names in data_dir/train
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


#zapisywanie do pliku nazw klas

classes_names_path = output_dir + '/classes_names.pkl'

classes_names_outfile = open(classes_names_path, 'wb')

pickle.dump(classes_names, classes_names_outfile)

classes_names_outfile.close()


#Sprawdzenie zawartosci pliku

# test_data_infile = open(classes_names_path, 'rb')
#
# print(pickle.load(test_data_infile))
#
# test_data_infile.close()


# BoW
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_data)

# .toarray(), bo X_train nie jest typu numpy array
X_train = X_train.toarray()

# .array, bo Y_train to lista a nie numpy array
# Y_train to ponumerowane klasy
Y_train = np.array(Y_train)

# scalenie Y_train i X_train w jednej macierzy
bow_train_data = np.c_[np.transpose(Y_train), X_train]

# zapisywanie do pliku - UWAGA: .pkl TO NIE JEST ROZSZERZENIE

train_data_path = output_dir + '/train_data.pkl'

train_data_outfile = open(train_data_path, 'wb')

pickle.dump(bow_train_data, train_data_outfile)

train_data_outfile.close()

# Sprawdzenie zawartosci pliku

# train_data_infile = open(train_data_path, 'rb')
#
# print(pickle.load(train_data_infile))
#
# train_data_infile.close()

# #classifier
# clf = MultinomialNB(alpha=.01)
# clf.fit(X_train,Y_train)


# --------------------------------------------------------
# read in test data

# X_test_data - wszystkie teksty z folderu test zawarate w jednej liście
# X_test - macierz czestotliwosci wystepowania slow

X_test_data, Y_test = read_data(input_dir + "/test/", classes_names)
X_test = vectorizer.transform(X_test_data)

X_test = X_test.toarray()
Y_test = np.array(Y_test)

# Scalenie Y_train i X_train do jednej macierzy

bow_test_data = np.c_[np.transpose(Y_test), X_test]

# zapisywanie do pliku - UWAGA: .pkl TO NIE JEST ROZSZERZENIE

test_data_path = output_dir + '/test_data.pkl'

test_data_outfile = open(test_data_path, 'wb')

pickle.dump(bow_test_data, test_data_outfile)

test_data_outfile.close()

# Sprawdzenie zawartosci pliku

# test_data_infile = open(test_data_path, 'rb')
#
# print(pickle.load(test_data_infile))
#
# test_data_infile.close()


# Klasyfikatory

# Y_pred=clf.predict(X_test)

# print("Accuracy: ", metrics.accuracy_score(Y_test,Y_pred))


