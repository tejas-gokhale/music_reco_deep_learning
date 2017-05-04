import string
import random
import numpy as np
import csv
import math    

def make_user_dict():
    user_dict = dict()
    user_song_dict = dict()
    with open('../data/pruned_triplets.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            user_id = row[0]
            song_id = row[1]
            plays   = row[2]
            if (user_id in user_dict):
                val = user_dict[user_id]
                val = val + [(song_id, plays)]
                user_dict[user_id] = val
                # user_song_dict[user_id] += [song_id]

            else:
                user_dict[user_id] = [(song_id, plays)]
                # user_song_dict[user_id] = [song_id]
            
            # if song_id in user_dict.values():
            #     user_song_dict[song_id] += [user_id]
            # else:
            #     user_song_dict[song_id] = [user_id]
            #     
    return user_dict
    
def make_song_dict():

    song_dict = dict()
    with open('../data/formatted_song_csv.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            song_no             = row[0]
            song_id				= row[1]
            album_id			= row[2]
            album_name			= row[3]
            artist_id 			= row[4]
            artist_latitude     = row[5]
            artist_location     = row[6]
            artist_longitude    = row[7]
            artist_name         = row[8]
            danceability        = row[9]
            duration            = row[10]
            key_signature       = row[11]
            key_signature_conf  = row[12]
            tempo               = row[13]
            time_signature      = row[14]
            time_signature_conf = row[15]
            title               = row[16]
            year                = row[17]
    
            song_dict[song_id] = [song_no, song_id, album_id, album_name, artist_id, artist_latitude, artist_location, artist_longitude, artist_name, danceability, duration, key_signature, key_signature_conf, tempo, time_signature, time_signature_conf, title, year]
    
    return song_dict

def make_user_profile(user_id, user_dict, song_dict):
    user_history = user_dict[user_id]

    artist_id_list = []
    duration_list  = []
    key_sig_list   = []
    tempo_list     = []
    time_sig_list  = []
    rating_list    = []

    for entry in user_history:
        song  = entry[0]
        plays = int(entry[1])
    
        song_profile = song_dict[song]
        artist_id 	 = str(song_profile[4])
        duration     = float(song_profile[10])*plays
        key_sig      = float(song_profile[11])*plays
        tempo        = float(song_profile[13])*plays
        time_sig     = float(song_profile[14])*plays
    
        artist_id_list.append(artist_id)
        duration_list.append(duration)
        key_sig_list.append(key_sig)
        tempo_list.append(tempo)
        time_sig_list.append(time_sig)
        rating_list.append(plays)

    avg_duration = sum(duration_list)/sum(rating_list)
    avg_key_sig  = sum(key_sig_list)/sum(rating_list)
    avg_tempo    = sum(tempo_list)/sum(rating_list)
    avg_time_sig = sum(time_sig_list)/sum(rating_list)

    return (avg_duration, avg_key_sig, avg_tempo, avg_time_sig, artist_id_list)

user_dict = make_user_dict()
song_dict = make_song_dict()

print ('1. user_dict and song_dict loaded')
"""
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

batch_size = 64
rand_idx = random.sample(range(len(data)), batch_size)
data_batch = data_np[rand_idx]
keys = range(5001)
bow_dict = [[]] * batch_size
for i in range(batch_size):
    # print(i)
    bow_dict[i] = {key : 1e-6 for key in keys}

print ('4. BOW initialized')
sep = ':'
for i in range(len(bow_dict)):
    data[i] = data[i].strip().split(',')
    print('#[%d / 64] ' % i, data[i][0])
    bow_dict[i][0] = corr_dict[data[i][0]]

    for j in range(2,len(data[i])):
        key = int(data[i][j].split(sep, 1)[0])
        val = int(data[i][j].split(sep, 1)[1])
        # print(data[i][j], key, val)
        bow_dict[i][key] = val
        
    bow_dict[i] = [(k, v) for k, v in bow_dict[i].items()]
    
    bow_dict[i] = [x[1] for x in bow_dict[i]]
    
    
    user_profile = make_user_profile(bow_dict[i][0], user_dict, song_dict)
    print (list(user_profile))
    # append the user profile for each user who has listened to this song
    # i.e. If 3 users listened to this song, we will have 3 training samples
    # bow_dict[i] = bow_dict[i] + list(user_profile)
"""