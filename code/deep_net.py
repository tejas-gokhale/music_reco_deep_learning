import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
batch_size_nn = 1
epochs = 20

train_set = np.genfromtxt('../data/data_train.csv', delimiter = ",")
test_set = np.genfromtxt('../data/data_test.csv', delimiter = ",")
print(train_set.shape)


input_nn = train_set[:, 0:100]
print(input_nn.shape)

input_normed = input_nn / input_nn.max(axis=0)
print(input_normed[1, :])

output_nn = train_set[:, -1]
output_normed = output_nn / output_nn.max(axis=0)
print(output_normed)

print(output_nn.shape)

def baseline_model():
	model = Sequential()
	#input_dim = (5004,)
	input_dim = (100,)
	model.add(InputLayer(input_shape=(input_nn.shape[1],)))
	model.add(Dense(50, activation='relu', input_shape = input_dim))
	model.add(Dropout(0.5))
	#model.add(Dense(20, activation='relu'))
	#model.add(Dropout(0.5))
	#model.add(Dense(10, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	print(model.summary())

	return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#model.fit(input_normed,output_normed,epochs = 10,batch_size=128, shuffle=True)
# test_normed = test_set[:, 0:100] / test_set[:, 0:100].max(axis=0)
# print("test_normed.shape",test_normed.shape)
# test_out_normed = test_set[:, -1] /test_set[:, -1].max(axis=0)
#score = model.evaluate(test_normed,test_out_normed, verbose=0)
#print("\n%s: %0.2f%%"%(model.metric_names[1]))
#print("score",score)
#y_pred = model.predict(test_normed,verbose=1)
#print("test_out_normed",test_out_normed)
#print("y_pred",y_pred)
#print("np.abs(y_pred - test_set[:, -1])/test_set[:, -1]",np.sum(np.abs(y_pred- test_out_normed)**2)/8657)

estimator = KerasRegressor(build_fn=model, nb_epoch = 20, batch_size = 16, verbose = 0, shuffle = True)

score = estimator.score(test_normed,test_out_normed)
print("score",score)