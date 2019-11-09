import argparse
import pickle
import numpy as np
from sklearn.decomposition import PCA

# przekazanie sciezek wejscia i wyjscia oraz wymiaru macierzy wyjscia
def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')
    parser.add_argument('--n-components', required=True, help='dimension', type=int)

    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.n_components

# przypisanie do zmiennych argumentow wymaganych przy wywolaniu pliku w konsoli
input_dir, output_dir, n = ParseArguments()

# otworzenie i zaladowanie plikow do zmiennych
train = open(input_dir + "train_data.pkl", "rb")
X_train = pickle.load(train)
train.close()

test = open(input_dir + "test_data.pkl", "rb")
X_test = pickle.load(test)
test.close()

# PCA
# uczenie/trenowanie PCA
pca = PCA(n_components=n, svd_solver='randomized')
# chcemy aby X_train byl typu numpy array
X_train = np.array(X_train)
# wybieramy kolumne z numerami klas
X_classes = X_train[::, 0]
# wycinamy kolumne z numerami klas
X_train = X_train[::, 1:]
# zastosowanie modelu pca na danych
X_train = pca.fit_transform(X_train)
# macierz z numerami klas laczymy ze zredukowana macierza
train_data = np.c_[X_classes, X_train]
# okreslamy sciezke wyjscia
train_data_path = output_dir + '/train_data.pkl'
# stworzenie pliku wyjscia
train_data_outfile = open(train_data_path, 'wb')
# zapisanie macierzy wyjscia do pliku wyjscia
pickle.dump(train_data, train_data_outfile)
# zamkniecie pliku wyjscia
train_data_outfile.close()

# analogicznie
X_test = np.array(X_test)
X_classes = X_test[::, 0]
X_test = X_test[::, 1:]
X_test = pca.fit_transform(X_test)
test_data = np.c_[X_classes, X_test]
test_data_path = output_dir + '/test_data.pkl'
test_data_outfile = open(test_data_path, 'wb')
pickle.dump(X_test, test_data_outfile)
test_data_outfile.close()
