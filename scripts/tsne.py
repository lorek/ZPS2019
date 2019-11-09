import argparse
import pickle
import numpy as np
from sklearn.manifold import TSNE


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')
    parser.add_argument('--n-components', required=True, help='dimension', type=int)

    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.n_components


input_dir, output_dir, n = ParseArguments()

train = open(input_dir + "train_data.pkl", "rb")
X_train = pickle.load(train)
train.close()

test = open(input_dir + "test_data.pkl", "rb")
X_test = pickle.load(test)
test.close()

# tSNE
tsne = TSNE(n_components=n)
X_train = np.array(X_train)
X_classes = X_train[::, 0]
X_train = X_train[::, 1:]
X_train = tsne.fit_transform(X_train)
train_data = np.c_[X_classes, X_train]
train_data_path = output_dir + '/train_data.pkl'
train_data_outfile = open(train_data_path, 'wb')
pickle.dump(train_data, train_data_outfile)
train_data_outfile.close()

X_test = np.array(X_test)
X_classes = X_test[::, 0]
X_test = X_test[::, 1:]
X_test = tsne.fit_transform(X_test)
test_data = np.c_[X_classes, X_test]
test_data_path = output_dir + '/test_data.pkl'
test_data_outfile = open(test_data_path, 'wb')
pickle.dump(X_test, test_data_outfile)
test_data_outfile.close()
