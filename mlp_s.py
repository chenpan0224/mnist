from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import time
from sklearn.externals import joblib

train_data_num = 60000 #The number of train figures
test_data_num = 10000 #The number of test figures
fig_w = 45       #width of each test figure

print('Loading data...')
train_data = np.load("mnist_train_s\mnist_train_data_s.npy")
train_label = np.fromfile("mnist_train\mnist_train_label",dtype=np.uint8)

test_data = np.load("mnist_test_s\mnist_test_data_s.npy")
test_label = np.fromfile("mnist_test\mnist_test_label",dtype=np.uint8)

x_train=train_data[0:60000, :]/255.0
y_train=train_label[0:60000]
x_test=test_data[0:10000, :]/255.0
y_test=test_label[0:10000]

print('Start training...')
time1=time.time()
clf=MLPClassifier(hidden_layer_sizes=(1000,), activation='logistic', solver='adam')
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

print('Predicting...')
y_predicted=clf.predict(x_test)
print(y_predicted)
print(classification_report(y_test, y_predicted))
time2=time.time()
print(time2-time1)