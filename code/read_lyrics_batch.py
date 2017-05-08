import string
import random
import numpy as np
import csv
import math 
from CNN_helper import cnn_helper
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, InputLayer
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras import backend as K
import numpy as np
import csv
batch_size_nn = 1
epochs = 20


#model = Sequential()
#input_dim = (5004,)
#input_dim = (100,)
#model.add(InputLayer(input_shape=(5004,)))
#model.add(Dense(50, activation='relu', input_shape = input_dim))
#model.add(Dense(20, activation='relu'))
#model.add(Dense(10, activation='relu'))
#model.add(Dense(1, activation='relu'))

# Compile model
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#print(model.summary())



####################
batch_size = 1000
iterations = 200 #CHANGE THIS
default_append = [239.236, 5.74, 126.20, 3.726, 2.68] #(avg_duration, avg_key_sig, avg_tempo, avg_time_sig, avg_rating )
####################


print ('1. cnn helper')
song_to_user_profile = cnn_helper()
print(len(song_to_user_profile.keys()))


with open('../data/mxm_dataset_train.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    data_np = np.array(data)
print ('2. Data loaded')


corr_tuples = []
with open('../data/unique_tracks.txt', 'r', encoding='utf-8') as f:
    song_track_corr = f.readlines()
    for line in song_track_corr:
        keyval = line.split('<SEP>')
        key = keyval[0]
        val = keyval[1]
        corr_tuples.append((key, val))
corr_dict = dict(corr_tuples)
print ('3. Correspondences loaded')


count = 0
num_keys = 5001
keys = range(num_keys)

flag = 0


with open('inputs.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for iteration in range(iterations):

        b_no     = batch_size*iteration 
        bow_dict = [[]] * batch_size

        for i in range(batch_size):
            bow_dict[i] = {key : 0 for key in keys}
        #count_b = 0
        #input_nn = np.zeros((64,100))
        #output_nn = np.zeros((64,1))
        for i in range(batch_size):
            data[i + b_no] = data[i + b_no].strip().split(',')
            print('#[%d] ' % (i  + b_no)) #PRINTING SERIAL NO.

            bow_dict[i][0] = corr_dict[data[i + b_no][0]]
            print(bow_dict[i][0]) #PRINTING SONG_ID

            inp    = ["0"]*num_keys

            for j in range(2, len(data[i + b_no])):
                key = int(data[i + b_no][j].split(':', 1)[0])
                val = int(data[i + b_no][j].split(':', 1)[1])
                bow_dict[i][key] = val
                inp[key] = str(val)

            if bow_dict[i][0] in song_to_user_profile:
                num_users = len(song_to_user_profile[bow_dict[i][0]])            

                if (num_users > 50):
                    num_users = 50


                for u in range(num_users):
                    line = inp[1:] + song_to_user_profile[bow_dict[i][0]][u][1:]
                    line = line[0:96]+line[-5:]

                    spamwriter.writerow(line)
     
print(count) #PRINTING COUNT





#input_nn[count_b,:] = line[0:96]+line[-5:-1]
#output_nn[count_b,:] = line[-1]
#count_b = count_b + 1
#count += 1
#print("line",line)

"""
Take line as input 
dimension : 5005 * 1
content   : (BOW, avg_duration, avg_key_sig, avg_tempo, avg_time_sig, label = avg_rating)
"""
#data_nn = line[0:5004]
#data_nn = line[0:96]+line[-5:-1]
#data_nn = np.array((data_nn))
#data_nn = np.expand_dims(data_nn,axis=0)
#print(data.shape)
#label = line[-1]
#label = np.array((label))
#label = np.expand_dims(label,axis=0)
#if count_b == 64: 
#   print("input_nn.shape",input_nn.shape)
  #  print("output_nn.shape",output_nn.shape)
   # model.fit(input_nn,output_nn,epochs = 50,batch_size=64)
   # count_b = 0


#score = model.evaluate(X,Y, verbose=0)
#print("\n%s: %0.2f%%"%(model.metric_names[1],score[1]*100))

        
        