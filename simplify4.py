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

train_data = train_data.reshape(train_data_num, fig_w, fig_w)
test_data = test_data.reshape(test_data_num, fig_w, fig_w)

train_data_s=np.zeros((60000, 144))
test_data_s=np.zeros((10000, 144))

train_data_s=train_data_s.reshape(60000, 12, 12)
test_data_s=test_data_s.reshape(10000, 12, 12)

def center(data, i, j):
	value=0
	for col in range(4):
		for row in range(4):
			value += data[i+col][j+row]
	return int(value/16)

for num in range(60000):
	for i in range(11):
		for j in range(11):
			train_data_s[num][i][j]=center(train_data[num], 4*i, 4*j)

for num in range(10000):
	for i in range(11):
		for j in range(11):
			test_data_s[num][i][j]=center(test_data[num], 4*i, 4*j)

for num in range(60000):
	for i in range(11):
		train_data_s[num][i][11]=train_data[num][i][11]
		train_data_s[num][11][i]=train_data[num][11][i]
	train_data_s[num][11][11]=train_data[num][11][11]

for num in range(10000):
	for i in range(11):
		test_data_s[num][i][11]=test_data[num][i][11]
		test_data_s[num][11][i]=test_data[num][11][i]
	test_data_s[num][11][11]=test_data[num][11][11]

train_data_s=train_data_s.reshape(60000, 144)
test_data_s=test_data_s.reshape(10000, 144)

np.save('mnist_train_s/mnist_train_data_s4', train_data_s)
np.save('mnist_test_s/mnist_test_data_s4', test_data_s)
