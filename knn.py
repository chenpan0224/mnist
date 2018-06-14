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
'''
pca=PCA(n_components=1000)
x_train=pca.fit_transform(x_train)
print(pca.explained_variance_ratio_.sum())
x_test=pca.fit_transform(x_test)
print(pca.explained_variance_ratio_.sum())
'''
'''
time1=time.time()
print('Start training...')
model = KNeighborsClassifier(5)
model.fit(x_train, y_train)

print('Start predicting...')
y_predicted = model.predict(x_test)
print (classification_report(y_test, y_predicted))
time2=time.time()
print(time2-time1)

print('Saving...')
joblib.dump(model, "knn_model.m")
'''
model=joblib.load('knn_model.m')
print('Start predicting...')
y_predicted = model.predict(x_test)
print (classification_report(y_test, y_predicted))