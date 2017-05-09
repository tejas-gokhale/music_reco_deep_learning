import numpy as np
import pandas
import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras import backend as K
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
import numpy.matlib as ml
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt


#####################
# Params
batch_size_nn = 1
num_epochs = 10
alpha = 0.1
input_size = 5004	# OPTIONS: 100 (top-100 BOW), 5004 (all BOW)
#####################

if input_size == 5004:
	train_set = np.genfromtxt('../data/all/data_train.csv', delimiter = ",")
	print('1. Read Train')
	test_set = np.genfromtxt('../data/all/data_test.csv', delimiter = ",")
	print('2. Read Test')
	valid_set = np.genfromtxt('../data/all/data_valid.csv', delimiter = ",")
	print('3. Read Validation')
elif input_size == 100:
	train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
	print('1. Read Train')
	test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
	print('2. Read Test')
	valid_set = np.genfromtxt('../data/data_valid.csv', delimiter = ",")
	print('3. Read Validation')



print(train_set.shape)

# TRAIN
train_in = train_set[:, 0:input_size]
print(train_in.shape)
train_in_normed = minmax_scale(train_in, axis = 0)
train_out = train_set[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in_normed = minmax_scale(test_set[:, 0:input_size], axis = 0)
test_out_normed = minmax_scale(test_set[:, -1] , axis = 0)

# VALIDATION
valid_in_normed = minmax_scale(valid_set[:, 0:input_size], axis = 0)
valid_out_normed = minmax_scale(valid_set[:, -1] , axis = 0)

# NETWORK : 100-50-20-1
model = Sequential()
input_dim = (input_size,)
#input_dim = (100,)
model.add(InputLayer(input_shape=(train_in.shape[1],)))
model.add(Dense(500, activation='relu', input_shape = input_dim, W_regularizer=regularizers.l2(alpha)))
model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu', W_regularizer=regularizers.l2(alpha)))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu', W_regularizer=regularizers.l2(alpha)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd')
print(model.summary())
    
# FIT MODEL ON TRAIN DATA

train_history = []
valid_history = []
test_history = []

for i in range(num_epochs):
	history = model.fit(train_in_normed, train_out_normed, epochs = 1, batch_size = 32, shuffle=True)
	train_history.append(list(history.history.values())[0][0])
	score_valid = model.evaluate(valid_in_normed, valid_out_normed, verbose=1)
	valid_history.append(score_valid)
	score_test = model.evaluate(test_in_normed, test_out_normed, verbose=1)
	test_history.append(score_test)

#print('\n')
#print(train_history)
#print(valid_history)
plt.gca().set_yscale('log')
plt.plot(range(1,num_epochs+1), train_history, label="train log-loss")
plt.plot(range(1,num_epochs+1), valid_history, label="validation log-loss")
plt.plot(range(1,num_epochs+1), test_history, label="test log-loss")
plt.legend()
plt.title('Training MSE Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()


# TEST ON TEST DATA
print('\n')
y_pred = model.predict(test_in_normed, verbose=1)
score_test = model.evaluate(test_in_normed, test_out_normed, verbose=1)
print ("######################")
print("score_test",score_test)
#print("y_pred.shape", y_pred.shape)
#print("test_out_normed.shape", test_out_normed.shape)
#print("y_pred", y_pred)

test_out_normed = np.expand_dims(test_out_normed, axis=1)
#print("test_out_normed", test_out_normed)


# print("y_pred",y_pred[0:10], "y_true", output_normed[0:10])
# print("np.abs(y_pred - y_true)/y_true)", np.sum(np.abs(y_pred - test_out_normed) **2) / y_pred.shape[0])