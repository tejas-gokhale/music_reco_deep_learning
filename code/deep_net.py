<<<<<<< HEAD
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
import numpy.matlib as ml
from sklearn.preprocessing import minmax_scale

batch_size_nn = 1
epochs = 20

train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
valid_set = np.genfromtxt('../data/data_valid.csv', delimiter = ",")

print(train_set.shape)

# TRAIN
train_in = train_set[:, 0:100]
print(train_in.shape)
train_in_normed = minmax_scale(train_in, axis = 0)
train_out = train_set[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in_normed = minmax_scale(test_set[:, 0:100], axis = 0)
test_out_normed = minmax_scale(test_set[:, -1] , axis = 0)

# VALIDATION
valid_in_normed = minmax_scale(valid_set[:, 0:100], axis = 0)
valid_out_normed = minmax_scale(valid_set[:, -1] , axis = 0)

model = Sequential()
#input_dim = (5004,)
input_dim = (100,)
model.add(InputLayer(input_shape=(train_in.shape[1],)))
model.add(Dense(300, activation='tanh', input_shape = input_dim))
#model.add(Dropout(0.5))
#model.add(Dropout(0.5))
# model.add(Dense(400, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dense(100, activation='tanh'))

#model.add(Dropout(0.5))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
    
# FUT MODEL ON TRAIN DATA
model.fit(train_in_normed, train_out_normed, epochs = 50, shuffle=True)

# EVALUATE ON VALIDATION DATA
score = model.evaluate(valid_in_normed, valid_out_normed, verbose=1)

# TEST ON TEST DATA
y_pred = model.predict(test_in_normed, verbose=1)

#print("\n%s: %0.2f%%"%(model.metric_names[1]))
print("score",score)
print("y_pred.shape", y_pred.shape)
print("y_pred", y_pred)
print("test_out_normed", test_out_normed)
test_out_normed = np.expand_dims(test_out_normed, axis=1)
#print("output_normed.shape", test_out_normed.shape)
print("test_out_normed.shape", test_out_normed.shape)

# print("y_pred",y_pred[0:10], "y_true", output_normed[0:10])
=======
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
import numpy.matlib as ml
from sklearn.preprocessing import minmax_scale

batch_size_nn = 1
epochs = 20

train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
valid_set = np.genfromtxt('../data/data_valid.csv', delimiter = ",")

print(train_set.shape)

# TRAIN
train_in = train_set[:, 0:100]
print(train_in.shape)
train_in_normed = minmax_scale(train_in, axis = 0)
train_out = train_set[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in_normed = minmax_scale(test_set[:, 0:100], axis = 0)
test_out_normed = minmax_scale(test_set[:, -1] , axis = 0)

# VALIDATION
valid_in_normed = minmax_scale(valid_set[:, 0:100], axis = 0)
valid_out_normed = minmax_scale(valid_set[:, -1] , axis = 0)

model = Sequential()
#input_dim = (5004,)
input_dim = (100,)
model.add(InputLayer(input_shape=(train_in.shape[1],)))
model.add(Dense(300, activation='tanh', input_shape = input_dim))
#model.add(Dropout(0.5))
#model.add(Dropout(0.5))
# model.add(Dense(400, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dense(100, activation='tanh'))

#model.add(Dropout(0.5))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
    
# FUT MODEL ON TRAIN DATA
model.fit(train_in_normed, train_out_normed, epochs = 50, shuffle=True)

# EVALUATE ON VALIDATION DATA
score = model.evaluate(valid_in_normed, valid_out_normed, verbose=1)

# TEST ON TEST DATA
y_pred = model.predict(test_in_normed, verbose=1)

#print("\n%s: %0.2f%%"%(model.metric_names[1]))
print("score",score)
print("y_pred.shape", y_pred.shape)
print("y_pred", y_pred)
print("test_out_normed", test_out_normed)
test_out_normed = np.expand_dims(test_out_normed, axis=1)
#print("output_normed.shape", test_out_normed.shape)
print("test_out_normed.shape", test_out_normed.shape)

# print("y_pred",y_pred[0:10], "y_true", output_normed[0:10])
>>>>>>> updated main
print("np.abs(y_pred - y_true)/y_true)", np.sum(np.abs(y_pred - test_out_normed) **2) / y_pred.shape[0])