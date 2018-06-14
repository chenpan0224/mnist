from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from sklearn.decomposition import PCA
import time

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
#train_label = train_label.reshape(train_data_num, 1)
#test_label = test_label.reshape(test_data_num, 1)
'''
clf=svm.SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovo')
clf.fit(train_data, train_label)
print (clf.score(test_data, test_label))
'''
x_train=train_data[0:60000, :]/255.0
y_train=train_label[0:60000]
x_test=test_data[0:10000, :]/255.0
y_test=test_label[0:10000]
'''
print('PCA...')
pca=PCA(n_components=500)
x_train=pca.fit_transform(x_train)
print(pca.explained_variance_ratio_.sum())
pca=PCA(n_components=500)
x_test=pca.fit_transform(x_test)
print(pca.explained_variance_ratio_.sum())
'''
print('Start training...')
time1=time.time()
clf=MLPClassifier(hidden_layer_sizes=(1000,), activation='logistic', solver='adam')
clf.fit(x_train, y_train)
time2=time.time()
print(clf.score(x_test, y_test), time2-time1)

print('Predicting...')
y_predicted=clf.predict(x_test)
print(y_predicted)
print(classification_report(y_test, y_predicted))
