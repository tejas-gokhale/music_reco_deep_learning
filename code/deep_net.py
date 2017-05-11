import numpy as np
#import pandas
import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras import backend as K
from keras import regularizers, optimizers
from keras.wrappers.scikit_learn import KerasRegressor
import numpy.matlib as ml
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from CNN_helper import generate_input_vector

#####################
# Params
batch_size_nn = 1
num_epochs = 10
alpha = 0.2
input_size = 100	# OPTIONS: 100 (top-100 BOW), 5004 (all BOW)
#input_size = 100
#####################

if input_size == 5004:
	train_set = np.genfromtxt('../data/all/data_train.csv', delimiter = ",")
	print('1. Read Train')
	np.random.shuffle(train_set)
	test_set = np.genfromtxt('../data/all/data_test.csv', delimiter = ",")
	print('2. Read Test')
	np.random.shuffle(test_set)
	valid_set = np.genfromtxt('../data/all/data_valid.csv', delimiter = ",")
	print('3. Read Validation')
	np.random.shuffle(valid_set)
elif input_size == 100:
	train_set = np.genfromtxt('../data/new/data_train.csv', delimiter = ",")
	print('1. Read Train')
	np.random.shuffle(train_set)
	print("train_set.shape",train_set.shape)
	test_set = np.genfromtxt('../data/new/data_test.csv', delimiter = ",")
	print('2. Read Test')
	np.random.shuffle(test_set)
	print("test_set.shape",test_set.shape)
	#np.savetxt("../data/test_shuffle.csv", test_set, delimiter=",", fmt='%s')
	valid_set = np.genfromtxt('../data/new/data_valid.csv', delimiter = ",")
	print('3. Read Validation')
	np.random.shuffle(valid_set)
	print("valid_set.shape",valid_set.shape)


# TRAIN
train_in = train_set[:, 0:input_size]
print(train_in.shape)
#train_in = transformer.fit_transform(train_set)
#train_in = train_in.toarray()

train_in_normed = minmax_scale(train_in, axis = 0)
train_out = train_set[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in = test_set[:, 0:input_size]
#test_in = transformer.fit_transform(test_set)
#test_in = test_in.toarray()
test_in_normed = minmax_scale(test_in, axis = 0)
test_out_normed = minmax_scale(test_set[:, -1] , axis = 0)

# VALIDATION
valid_in = valid_set[:, 0:input_size]
#valid_in = transformer.fit_transform(valid_set)
#valid_in = valid_in.toarray()
valid_in_normed = minmax_scale(valid_set[:, 0:input_size], axis = 0)
valid_out_normed = minmax_scale(valid_set[:, -1] , axis = 0)

# NETWORK : 100-50-20-1
model = Sequential()
input_dim = (input_size,)
#input_dim = (100,)
model.add(InputLayer(input_shape=(train_in.shape[1],)))
model.add(Dense(500, activation='tanh', input_shape = input_dim, W_regularizer=regularizers.l2(alpha)))
#model.add(Dense(50, activation='tanh', input_shape = input_dim, W_regularizer=regularizers.l2(alpha)))
#model.add(Dropout(0.5))
model.add(Dense(200, activation='tanh', W_regularizer=regularizers.l2(alpha)))
#model.add(Dense(200, activation='tanh'))
#model.add(Dense(20, activation='tanh',W_regularizer=regularizers.l2(alpha)))
#model.add(Dropout(0.5))
model.add(Dense(100, activation='tanh', W_regularizer=regularizers.l2(alpha)))
#model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='tanh',W_regularizer=regularizers.l2(alpha)))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
#sgd = Keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
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

my_user = '8305c896f42308824da7d4386f4b9ee584281412'
input_vector = generate_input_vector(my_user)

network_input = input_vector[:,1:5005]
network_input_normed = minmax_scale(network_input_normed, axis = 0)
song_ratings = model.predict(network_input_normed, verbose = 1)

print(input_vector[np.argmax(song_ratings)][0])

#with open('ratings.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter='', quotechar='|', quoting=csv.QUOTE_MINIMAL)

#    for entry in song_ratings:
#        spamwriter.writerow(entry)

   # for entry in encoded_test:
   #     spamwriter.writerow(entry)
