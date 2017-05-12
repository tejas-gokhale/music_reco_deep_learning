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
import matplotlib
matplotlib.use('Agg')
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

<<<<<<< HEAD
if input_size == 1004:
	train_set = np.genfromtxt('../data/all/data_train.csv', delimiter = ",")
=======
if input_size == 5004:
	train_set = np.genfromtxt('../data/new/data_train.csv', delimiter = ",")
>>>>>>> 3bc9c40d564bf2be6fc17a307b47dafbefd0488b
	print('1. Read Train')
	np.random.shuffle(train_set)
	test_set = np.genfromtxt('../data/new/data_test.csv', delimiter = ",")
	print('2. Read Test')
	np.random.shuffle(test_set)
	valid_set = np.genfromtxt('../data/new/data_valid.csv', delimiter = ",")
	print('3. Read Validation')
	np.random.shuffle(valid_set)
elif input_size == 100:
	train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
	print('1. Read Train')
	np.random.shuffle(train_set)
	print("train_set.shape",train_set.shape)
	test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
	print('2. Read Test')
	np.random.shuffle(test_set)
	print("test_set.shape",test_set.shape)
	#np.savetxt("../data/test_shuffle.csv", test_set, delimiter=",", fmt='%s')
	valid_set = np.genfromtxt('../data/data_valid.csv', delimiter = ",")
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
plt.plot(range(1,num_epochs+1), train_history, label="train loss")
plt.plot(range(1,num_epochs+1), valid_history, label="validation loss")
plt.plot(range(1,num_epochs+1), test_history, label="test loss")
plt.legend()
plt.title('Training MSE Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
#plt.show()
fig = plt.figure()
fig.savefig('temp.png')


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

my_user = '5a905f000fc1ff3df7ca807d57edb608863db05d'
input_vector = generate_input_vector(my_user)
print (input_vector[0])

network_input = []
for i in range(len(input_vector)):
	network_input.append(input_vector[i][1:101])
print (network_input[0])


<<<<<<< HEAD
network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])
 
=======
>>>>>>> 3bc9c40d564bf2be6fc17a307b47dafbefd0488b
network_input_normed = minmax_scale(np.array(network_input), axis = 0)
song_ratings = model.predict(network_input_normed, verbose = 1)
 
print("1",input_vector[np.argmax(song_ratings)][0])

my_user = '732f88be38fae217f8ab7e24c20dd072436e3e40'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
song_ratings = model.predict(network_input_normed, verbose = 1)

print("2",input_vector[np.argmax(song_ratings)][0])

my_user = '9b887e10a4711486085c4fae2d2599fc0d2c484d'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
song_ratings = model.predict(network_input_normed, verbose = 1)

print("3",input_vector[np.argmax(song_ratings)][0])

my_user = '76235885b32c4e8c82760c340dc54f9b608d7d7e'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
song_ratings = model.predict(network_input_normed, verbose = 1)

print("4",input_vector[np.argmax(song_ratings)][0])


#network_input = input_vector[:,1:5005]
#network_input_normed = minmax_scale(network_ut, axis = 0)
#song_ratings = model.predict(network_input_normed, verbose = 1)

#print("5a905f000fc1ff3df7ca807d57edb608863db05d",input_vector[np.argmax(song_ratings)][0])

#my_user_1 = '732f88be38fae217f8ab7e24c20dd072436e3e40'
#input_vector_1 = np.array(generate_input_vector(my_user_1))

#network_input_1 = input_vector_1[:,1:5005]
#network_input_normed_1 = minmax_scale(network_input_1, axis = 0)
#song_ratings_1 = model.predict(network_input_normed_1, verbose = 1)
#print("732f88be38fae217f8ab7e24c20dd072436e3e40",input_vector_1[np.argmax(song_ratings_1)][0])

#my_user_2 = '9b887e10a4711486085c4fae2d2599fc0d2c484d'
#input_vector_2 = np.array(generate_input_vector(my_user_2))

#network_input_2 = input_vector_2[:,1:5005]
#network_input_normed_2 = minmax_scale(network_input_2, axis = 0)
#song_ratings_2 = model.predict(network_input_normed_2, verbose = 1)
#print("9b88",input_vector_2[np.argmax(song_ratings_2)][0])

#my_user_3 = '76235885b32c4e8c82760c340dc54f9b608d7d7e' 
#input_vector_3 = np.array(generate_input_vector(my_user_3))

#network_input_3 = input_vector_3[:,1:5005]
#network_input_normed_3 = minmax_scale(network_input_3, axis = 0)
#song_ratings_3 = model.predict(network_input_normed_3, verbose = 1)
#print("762",input_vector_3[np.argmax(song_ratings_3)][0])




# with open('ratings.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter='', quotechar='|', quoting=csv.QUOTE_MINIMAL)

#    for entry in song_ratings:
#        spamwriter.writerow(entry)

#    for entry in encoded_test:
#        spamwriter.writerow(entry)
