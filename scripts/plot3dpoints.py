from sklearn import svm
import pickle
import argparse

#
 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd


fig = plt.figure()
#to bedzie 3d
ax = fig.add_subplot(111, projection='3d')

#ile punktow
n = 50
 
 
 
 



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
X_test,Y_test = points_train[:,1:],points_train[:,0]

cl0=np.argwhere(Y_train==0)
cl1=np.argwhere(Y_train==1)
cl2=np.argwhere(Y_train==2)

ax.scatter(X_train[cl0][:,0][:,0], X_train[cl0][:,1][:,1],X_train[cl0][:,2][:,2],c='r')
ax.scatter(X_train[cl1][:,0][:,0], X_train[cl1][:,1][:,1],X_train[cl1][:,2][:,2],c='g')
ax.scatter(X_train[cl2][:,0][:,0], X_train[cl2][:,1][:,1],X_train[cl2][:,2][:,2],c='b')


#ax.scatter(xs, ys, zs, c=c, marker=m)
 
 

# ~ for c, m  in [('r', 'o' ), ('b', '^' )]:
    # ~ xs = rnd.rand(n)*20-15
    # ~ ys = rnd.rand(n)*30-20
    # ~ zs = rnd.rand(n)*40-20
    
plt.title('')
plt.show()

quit()


