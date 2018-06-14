from sklearn import svm
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from sklearn.decomposition import PCA
import time
from sklearn.externals import joblib

train_data_num = 60000 #The number of train figures
test_data_num = 10000 #The number of test figures
fig_w = 45       #width of each test figure

print('Loading data...')
train_data = np.fromfile("mnist_train\mnist_train_data",dtype=np.uint8)
train_label = np.fromfile("mnist_train\mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test\mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist_test\mnist_test_label",dtype=np.uint8)

train_data = train_data.reshape(train_data_num, fig_w*fig_w)
test_data = test_data.reshape(test_data_num, fig_w*fig_w)

x_train=train_data[0:60000, :]/255.0
y_train=train_label[0:60000]
x_test=test_data[0:10000, :]/255.0
y_test=test_label[0:10000]

print('Start training...')
time1=time.time()
clf=svm.SVC(C=5, kernel='rbf', gamma=0.05, decision_function_shape='ovr')
clf.fit(x_train, y_train)
time2=time.time()

print('Saving...')
joblib.dump(clf, "svm_model.m")