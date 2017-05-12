from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.preprocessing import minmax_scale
from CNN_helper import generate_input_vector

#encoding_dim = 100
encoding_dim = 100

print("starting")
x_train = np.genfromtxt('../data/ae/data_train_ae.csv', delimiter = ",")
print('1. Read Train')
np.random.shuffle(x_train)
x_test = np.genfromtxt('../data/ae/data_test_ae.csv', delimiter = ",")
print('2. Read Test')
np.random.shuffle(x_test)
#valid_set = np.genfromtxt('../data/all/data_valid.csv', delimiter = ",")
#print('3. Read Validation')
#np.random.shuffle(valid_set)
train_in = x_train[:, 0:5004]
print(train_in.shape)
train_in_normed = minmax_scale(train_in, axis = 0)
train_out = x_train[:, -1]
train_out_normed = minmax_scale(train_out, axis = 0)

# TEST
test_in_normed = minmax_scale(x_test[:, 0:5004], axis = 0)
test_out_normed = minmax_scale(x_test[:, -1] , axis = 0)


input_size = Input(shape=(5004,))
encoded = Dense(1000, activation='relu')(input_size)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(200, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)

#decoded = Dense(200, activation='relu')(encoded)
decoded = Dense(200, activation='relu')(encoded)
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
#decoder = Model(encoded_input, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train_in_normed, train_in_normed,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(test_in_normed, test_in_normed))

encoded_train = encoder.predict(train_in_normed)
encoded_test = encoder.predict(test_in_normed)
#print(encoded_input)
#decoded_imgs = decoder.predict(encoded_imgs)

my_user = '5a905f000fc1ff3df7ca807d57edb608863db05d'
input_vector = generate_input_vector(my_user)
print (input_vector[0])

network_input = []
for i in range(len(input_vector)):
	network_input.append(input_vector[i][1:101])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
encoded_1 = encoder.predict(network_input_normed, verbose = 1)

with open('encoded_1.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for entry in encoded_1:
        spamwriter.writerow(entry)
 
#print("1",input_vector[np.argmax(song_ratings)][0])

my_user = '732f88be38fae217f8ab7e24c20dd072436e3e40'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
encoded_2 = encoder.predict(network_input_normed, verbose = 1)

with open('encoded_2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for entry in encoded_train:
        spamwriter.writerow(entry)



#print("2",input_vector[np.argmax(song_ratings)][0])

my_user = '9b887e10a4711486085c4fae2d2599fc0d2c484d'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
encoded_3 = encoder.predict(network_input_normed, verbose = 1)

with open('encoded_3.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for entry in encoded_train:
        spamwriter.writerow(entry)
#print("3",input_vector[np.argmax(song_ratings)][0])

my_user = '76235885b32c4e8c82760c340dc54f9b608d7d7e'
input_vector = generate_input_vector(my_user)

network_input=[]

for i in range(len(input_vector)):
              network_input.append(input_vector[i][1:])
#print (network_input[0])

network_input_normed = minmax_scale(np.array(network_input), axis = 0)
encoded_4 = encoder.predict(network_input_normed, verbose = 1)

with open('encoded_4.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for entry in encoded_train:
        spamwriter.writerow(entry)
#print("4",input_vector[np.argmax(song_ratings)][0])
