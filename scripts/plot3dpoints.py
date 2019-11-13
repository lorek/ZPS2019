from sklearn import svm
import pickle
import argparse
import os


#
 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
  
 



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


if not os.path.isfile(input_dir +'/classes_names.pkl' ):
    print("The following file is missing:  "+input_dir+"classes_names.pkl" +" !!!")
    quit()

infile_classes = open( input_dir +'/classes_names.pkl' , 'rb')
classes_names = pickle.load(infile_classes)
infile_classes.close()

X_train,Y_train = points_train[:,1:],points_train[:,0]
X_test,Y_test = points_test[:,1:],points_test[:,0]


if X_train.shape[1]!=3:
    print("These are not 3d points !!! (they are "+str(X_train.shape[1])+"-dimensional)")
    quit()

#osobne rysunki dla train i test

#TRAIN
fig_train = plt.figure(1)
ax_train = fig_train.add_subplot(111, projection='3d')

ax_train.set_title(input_dir +" - TRAIN")	

   


for cl in range(0,int(Y_train.max())+1):
    points = X_train[Y_train==cl];
    ax_train.scatter(points[:,0], points[:,1], points[:,2], label=classes_names[cl])
    
ax_train.legend()
 	   
    
# TEST   (na razie nie dziala) 
    
# ~ fig_test = plt.figure(2)
# ~ ax_test = fig_test.add_subplot(111, projection='3d')

# ~ ax_test.set_title(input_dir +" - TRAIN")	

# ~ for cl in range(0,int(Y_test.max())+1):
    # ~ points = X_test[Y_test==cl];
    # ~ ax_test.scatter(points[:,0], points[:,1], points[:,2], label=classes_names[cl])
    
# ~ ax_test.legend()
 	   
    
 
 
plt.show()

quit()


