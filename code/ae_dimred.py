from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

## TRAIN DATA
## -------------
train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
print(train_set.shape)
X_train = train_set[:, 0:-1]
X_train = minmax_scale(X_train, axis = 0)
Y_train = train_set[:, -1]

print(X_train[0, :])

ncol = train_set.shape[1]
print (ncol)

## TEST DATA
## ------------
test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
print(test_set.shape)
X_test = test_set[:, 0:-1]
X_test = minmax_scale(X_test, axis = 0)
Y_test = test_set[:, -1]
print(X_test[0, :])

## NETWORK
## --------
input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 100
# DEFINE THE ENCODER LAYERS
encoded1 = Dense(1000, activation = 'relu')(input_dim)
encoded2 = Dense(500, activation = 'relu')(encoded1)
encoded3 = Dense(200, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)
# DEFINE THE DECODER LAYERS
decoded1 = Dense(200, activation = 'relu')(encoded4)
decoded2 = Dense(500, activation = 'relu')(decoded1)
decoded3 = Dense(1000, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded4)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch = 100, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input = input_dim, output = encoded4)
encoded_input = Input(shape = (encoding_dim, ))
# encoded_out = encoder.predict(X_test)
# encoded_out[0:2]
"""