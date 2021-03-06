
import pickle
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')

    args = parser.parse_args()

    return (args.input_dir, args.output_dir)

input_dir, output_dir= ParseArguments()

train = open(input_dir + "train_data.pkl", "rb")
pointsTrain = pickle.load(train)
train.close()

test = open(input_dir + "test_data.pkl", "rb")
pointsTest = pickle.load(test)
test.close()


XTrain, YTrain = pointsTrain[::,1:],pointsTrain[::,0]
XTest, YTest = pointsTest[::,1:],pointsTest[::,0]

knn=5
knnClassifier = KNeighborsClassifier(n_neighbors=knn)
knnClassifier.fit(XTrain, YTrain)
prediction = knnClassifier.predict(XTest)
print("Accurancy:", metrics.accuracy_score(YTest, prediction))



