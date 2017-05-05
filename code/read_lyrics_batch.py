import string
import random
import numpy as np
import csv
import math 
from CNN_helper import cnn_helper

print ('1. cnn helper')
song_to_user_profile = cnn_helper()
print(len(song_to_user_profile.keys()))

with open('../data/mxm_dataset_train_chhotu.txt', 'r', encoding='utf-8') as f:
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

pruning_list = corr_dict.keys()
# print (list(pruning_list)[0], data_np[0].split(',')[0])
# print (len(pruning_list))
# idx_remove = list()
# for i in range(len(data_np)):
#     if corr_dict[data_np[i].split(',')[0]] in song_to_user_profile.keys():
#         idx_remove.append(i)
#         # print(i)
# 
# print(len(idx_remove))
    

print ('3. Correspondences loaded')

batch_size = 64 # default 64
rand_idx = random.sample(range(len(data)), batch_size)
data_batch = data_np[rand_idx]
keys = range(5001)
bow_dict = [[]] * batch_size
default_append = [239.236, 5.74, 3.726, 126.20]

for i in range(batch_size):
    # print(i)
    bow_dict[i] = {key : 1e-6 for key in keys}

print ('4. BOW initialized')
sep = ':'
i = 0
for i in range(64):
    data[i] = data[i].strip().split(',')
    print('#[%d / 64] ' % i, data[i][0])
    bow_dict[i][0] = corr_dict[data[i][0]]
    print(bow_dict[i][0])

    for j in range(2,len(data[i])):
        key = int(data[i][j].split(sep, 1)[0])
        val = int(data[i][j].split(sep, 1)[1])
        # print(data[i][j], key, val)
        bow_dict[i][key] = val
        
    bow_dict[i] = [(k, v) for k, v in bow_dict[i].items()]
    
    bow_dict[i] = [x[1] for x in bow_dict[i]]
    
    print(len(bow_dict[i]))
    
    if bow_dict[i][0] in song_to_user_profile:
        i = i+1
        print(bow_dict[i][0])
        num_users = len(song_to_user_profile[bow_dict[i][0]])
        inp = [[]] * num_users
        for u in range(num_users):
            # print (song_to_user_profile[bow_dict[i][0]])
            #bow_dict[i] = bow_dict[i] + default_append
            line = bow_dict[i] + default_append
            inp[u] = line
            # bow_dict[i].append(song_to_user_profile[bow_dict[i][0]])
    else:
        inp = bow_dict[i] + default_append
    
    print(len(inp))
print(i)
        
        
    # user_profile = make_user_profile(bow_dict[i][0], user_dict, song_dict)
    # append the user profile for each user who has listened to this song
    # i.e. If 3 users listened to this song, we will have 3 training samples
    # bow_dict[i] = bow_dict[i] + list(user_profile)