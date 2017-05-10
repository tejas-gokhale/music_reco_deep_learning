from keras.layers import Input, Dense
from keras.models import Model

#encoding_dim = 100
encoding_dim = 20

x_train = np.genfromtxt('../data/all/data_train_ae.csv', delimiter = ",")
print('1. Read Train')
np.random.shuffle(train_set)
x_test = np.genfromtxt('../data/all/data_test_ae.csv', delimiter = ",")
print('2. Read Test')
np.random.shuffle(test_set)
#valid_set = np.genfromtxt('../data/all/data_valid.csv', delimiter = ",")
#print('3. Read Validation')
#np.random.shuffle(valid_set)
train_in = x_train[:, 0:5004]
print(train_in.shape)
train_in_normed = minmax_scale(train_in, axis = 0)
train_out = x_train[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in_normed = minmax_scale(x_test[:, 0:input_size], axis = 0)
test_out_normed = minmax_scale(x_test[:, -1] , axis = 0)


input_size = Input(shape=(5004,))
encoded = Dense(1000, activation='relu')(input_size)
encoded = Dense(500, activation='relu')(encoded)
#encoded = Dense(200, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)

#decoded = Dense(200, activation='relu')(encoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(5004, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_size, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_size, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train_in_normed, train_in_normed,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(train_out_normed, train_out_normed))

encoded_input = encoder.predict(train_out_normed)
#decoded_imgs = decoder.predict(encoded_imgs)