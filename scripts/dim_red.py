import argparse
import pickle
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA, LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE
from shutil import copy2

# przekazanie sciezek wejscia i wyjscia oraz wymiaru macierzy wyjscia
def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=True, help='output dir')
    parser.add_argument('--method', required=True, help='method: pca, kernelpca, nmf, ica, tsne, lda', type=str)
    parser.add_argument('--n-components', required=True, help='dimension', type=int)

    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.method, args.n_components

# przypisanie do zmiennych argumentow wymaganych przy wywolaniu pliku w konsoli
input_dir, output_dir, method, n = ParseArguments()

# skopiowanie pliku z nazwami klas
outfile = open(output_dir + "/classes_names.pkl", 'wb')
copy2(input_dir + "classes_names.pkl", output_dir + "/classes_names.pkl")
outfile.close()
# otworzenie i zaladowanie plikow do zmiennych
train = open(input_dir + "train_data.pkl", "rb")
X_train = pickle.load(train)
train.close()

test = open(input_dir + "test_data.pkl", "rb")
X_test = pickle.load(test)
test.close()
if method == 'tsne':
    print('Method : tSNE')
    # zdefiniowanie metody
    m = TSNE(n_components=n)
    # chcemy aby X_train i X_test byly typu numpy array;
    # wybieramy odpowiednie kolumny z numerami klas;
    # wybieramy kolumny z danymi
    X_train = np.array(X_train)
    X_classes_train = X_train[::, 0]
    X_train = X_train[::, 1:]
    X_test = np.array(X_test)
    X_classes_test = X_test[::, 0]
    X_test = X_test[::, 1:]
    # laczymy macierze X_train i X_test w jedna
    train_matrix = np.zeros((X_train.shape[0] + X_test.shape[0], X_train.shape[1]))
    train_matrix[:X_train.shape[0], ::] = X_train
    train_matrix[X_train.shape[0]:, ::] = X_test
    # trenujemy metoda na polaczonej macierzy
    train_matrix = m.fit_transform(train_matrix)
    # wybieramy dane ktore nalezaly do danych treningowych
    X_train = train_matrix[:X_train.shape[0], ::]
    # laczymy macierz z numerami klas
    train_data = np.c_[X_classes_train, X_train]
    # podajemy sciezke wyjscia dla X_train
    train_data_path = output_dir + '/train_data.pkl'
    # zapisujemy X_train
    train_data_outfile = open(train_data_path, 'wb')
    pickle.dump(train_data, train_data_outfile)
    train_data_outfile.close()
    # analogicznie jak dla X_train
    X_test = train_matrix[X_train.shape[0]:, ::]
    test_data = np.c_[X_classes_test, X_test]
    test_data_path = output_dir + '/test_data.pkl'
    test_data_outfile = open(test_data_path, 'wb')
    pickle.dump(test_data, test_data_outfile)
    test_data_outfile.close()
    print('Done', train_data.shape, test_data.shape)
else:
    if method == 'pca':
        m = PCA(n_components=n)
        print('Method : PCA')
    if method == 'kernelpca':
        # kernel='poly' wybieramy nieliniowe jadro
        m = KernelPCA(n_components=n, kernel='poly')
        print('Method : KernelPCA')
    if method == 'nmf':
        m = NMF(n_components=n)
        print('Method : NMF')
    if method == 'ica':
        m = FastICA(n_components=n)
        print('Method : ICA')
    if method == 'lda':
        m = LDA(n_components=n)
        print('Method : LDA')
    # uczenie/trenowanie metody
    # chcemy aby X_train byl typu numpy array
    X_train = np.array(X_train)
    # wybieramy kolumne z numerami klas
    X_classes = X_train[::, 0]
    # wycinamy kolumne z numerami klas
    X_train = X_train[::, 1:]
    # zastosowanie modelu pca na danych
    X_train = m.fit_transform(X_train)
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
    # chcemy aby X_test był typu numpy array
    X_test = np.array(X_test)
    # wybieramy kolumne z numerami klas
    X_classes = X_test[::, 0]
    # wycinamy kolumne z numerami klas
    X_test = X_test[::, 1:]
    # wykorzystanie wytrenowanej tranformacji na macierzy danych testowych
    X_test = m.transform(X_test)
    # analogicznie jak X_train
    test_data = np.c_[X_classes, X_test]
    test_data_path = output_dir + '/test_data.pkl'
    test_data_outfile = open(test_data_path, 'wb')
    pickle.dump(test_data, test_data_outfile)
    test_data_outfile.close()
    print('Done', train_data.shape, test_data.shape)

