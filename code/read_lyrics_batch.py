import string
import random
import numpy as np
import csv
import math 
from CNN_helper import cnn_helper
import numpy as np
import csv
batch_size_nn = 1
epochs = 20

####################
batch_size = 1000
iterations = 100 #CHANGE THIS
default_append = [239.236, 5.74, 126.20, 3.726, 2.68] #(avg_duration, avg_key_sig, avg_tempo, avg_time_sig, avg_rating )
####################

## LOAD HELPER
print ('1. cnn helper')
song_to_user_profile = cnn_helper()
print(len(song_to_user_profile.keys()))

## LOAD DATA
with open('../data/mxm_dataset_train.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    data_np = np.array(data)
print ('2. Data loaded')

## LOAD CORRESPONDENCES
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


## WRITE TO FILE
count = 0
num_keys = 5001
keys = range(num_keys)

flag = 0


with open('../data/inputs.csv', 'w', newline='') as csvfile:
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
                    #line = line[0:96]+line[-5:]

                    spamwriter.writerow(line)
     
print(count) #PRINTING COUNT