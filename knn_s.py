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
train_data = np.load("mnist_train_s\mnist_train_data_s4.npy")
train_label = np.fromfile("mnist_train\mnist_train_label",dtype=np.uint8)

test_data = np.load("mnist_test_s\mnist_test_data_s4.npy")
test_label = np.fromfile("mnist_test\mnist_test_label",dtype=np.uint8)

x_train=train_data[0:60000, :]/255.0
y_train=train_label[0:60000]
x_test=test_data[0:10000, :]/255.0
y_test=test_label[0:10000]

time1=time.time()
print('Start training...')
model = KNeighborsClassifier(5)
model.fit(x_train, y_train)

print('Start predicting...')
y_predicted = model.predict(x_test)
print (classification_report(y_test, y_predicted))
time2=time.time()
print(time2-time1)