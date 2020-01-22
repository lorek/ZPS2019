from sklearn.ensemble import AdaBoostClassifier
import pickle
import argparse

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--input-dir', default="", required=True, help='data dir')
    parser.add_argument('--output-dir', default="", required=False, help='output dir')

    args = parser.parse_args()

    return (args.input_dir, args.output_dir)

input_dir, output_dir= ParseArguments()

infile = open(input_dir + "/train_data.pkl", 'rb')
points_train= pickle.load(infile)
infile.close()


infile = open( input_dir +'/test_data.pkl' , 'rb')
points_test= pickle.load(infile)
infile.close()


X_train,Y_train = points_train[:,1:],points_train[:,0]
X_test,Y_test = points_test[:,1:],points_test[:,0]
clf = AdaBoostClassifier()
clf.fit(X_train, Y_train)
classes_predicted = clf.predict(X_test)

rate = 0;
nr = 0;
for cl_p, cl_v in zip(classes_predicted, Y_test):
	if (cl_p == cl_v):
		rate = rate + 1
	nr = nr + 1

rate2 = rate / len(classes_predicted)
print("classification rate = ", rate2)