## Recommender Systems
# Author:   Tejas Gokhale
# Date:     20-MAR-2017

## imports
import os
import numpy as np
import time
import operator
import math
import csv
import random
import matplotlib.pyplot as plt

## Load Database
# F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

datapath = os.path.join('../data/', 'triplets.txt')
def NoneToZero(dic):
    for key,val in dic.items():
        if val==None:
            dic[key] = 0.0
    return dic

def AverageDictValue(dic):
    avg_val = sum(dic.values())/len(dic)
    return avg_val
    
def loadMSD():
    with open(datapath, 'r', encoding="utf8") as f:
        datalines = f.readlines()
    prefs = {}
    for line in datalines:
        (uid, mid, rating) = line.split('\t')
        prefs.setdefault(uid, {})
        prefs[uid][mid] = float(rating)
    return prefs


## GLOBAL VARIABLES
# --------------------------
data = loadMSD()

print ('Data Loaded')

uid_list = [d for d in data]

print (len(uid_list))

##
         
def pearsonSimilarity(u_a, u_b):
    ratings_a = data[u_a]
    ratings_b = data[u_b]
    
    avg_ratings_a = sum(ratings_a.values())/len(ratings_a)
    avg_ratings_b = sum(ratings_b.values())/len(ratings_b)
    
    # print(avg_ratings_a)
    # print(avg_ratings_b)
    
    set_a = set(ratings_a)
    set_b = set(ratings_b)
    
    m = list()
    term0 = 0
    term1 = 0
    term2 = 0
    for movie in set_a.intersection(set_b):
        m.append(movie)
        term0 = term0 + (ratings_a[movie] - avg_ratings_a)*(ratings_b[movie] - avg_ratings_b)
        term1 = term1 + (ratings_a[movie] - avg_ratings_a)**2
        term2 = term2 + (ratings_b[movie] - avg_ratings_b)**2

    
    if term1 !=0:
        if term2 != 0:
            if term0 <= 0:
                pearson_similarity = -1000
            else:
                pearson_similarity = math.log(term0) - (math.log(np.sqrt(term1)) + math.log(term2))
        else:
            pearson_similarity = 0
    else:
        pearson_similarity = 0
        
    #print (u_b,term0, term1, term2, pearson_similarity)
    return set_a, set_b, pearson_similarity
##

def makeRecommendation(my_user):
    songs_my = [k for k,v in data[my_user].items()]
    
    sim = dict()
    for user in uid_list:
        set_my, set_user, sim[user] = pearsonSimilarity(my_user, user)
    # print(sim['5a905f000fc1ff3df7ca807d57edb608863db05d'])
    # print(max(sim.items(), key=operator.itemgetter(1))[0])
    # print (sim)
    top10 = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)[:10]
    
    # print(top10)
    songs_users = list()
    for user, val in top10:
        songs_users = songs_users + [k for k,v in data[user].items()]
    
    print(len(songs_users))
    songs_users = list(set(songs_users))
    print(len(songs_users))
    songs_users_not_me = [x for x in songs_users if x not in songs_my]
    print(len(songs_users_not_me))
    
    candidates = NoneToZero(dict.fromkeys(songs_users_not_me))
    for user, weight in top10:
        songs_this_user = [k for k,v in data[user].items()]
        for song in songs_users_not_me:
            if song in songs_this_user:
                plays = data[user][song] - AverageDictValue(data[user])
                candidates[song] = candidates[song] + weight * plays
    
    # print(candidates)
    # print (len(candidates))
    print(max(candidates, key=candidates.get))
    return top10

my_user = '8305c896f42308824da7d4386f4b9ee584281412'
top10 = makeRecommendation(my_user)


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
                user_song_dict[user_id] += [song_id]
    
            else:
                user_dict[user_id] = [(song_id, plays)]
                user_song_dict[user_id] = [song_id]
    
    
    return user_dict, user_song_dict


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


song_dict = make_song_dict()
user_dict, user_song_dict = make_user_dict()

avg_dur_array = []
avg_key_array = []
avg_tempo_array = []
avg_time_array = []
num_of_common_artists = []

for iter in range(500):
    my_user_id = random.choice(list(user_dict))
    # print (user_id)
    my_user_profile = make_user_profile(my_user_id, user_dict, song_dict)
    
    bhai_user_profile = make_user_profile(top10[0][0], user_dict, song_dict)
    
    num_of_common_artists.append(100.0 * len(set(my_user_profile[4]).intersection(bhai_user_profile[4])) / len(my_user_profile[4]))
    
    
    avg_dur_array.append(100.0 *abs(my_user_profile[0] - bhai_user_profile[0])/(my_user_profile[0] + + 10e-4))
    avg_key_array.append(100.0 *abs(my_user_profile[1] - bhai_user_profile[1])/(my_user_profile[1] + 10e-4))
    avg_tempo_array.append(100.0 *abs(my_user_profile[2] - bhai_user_profile[2])/(my_user_profile[2] + 10e-4))
    avg_time_array.append(100.0 *abs(my_user_profile[3] - bhai_user_profile[3])/(my_user_profile[3] + + 10e-4))
    
    if iter % 100 == 0:
        print (iter)        

plt.subplots_adjust(hspace=0.5)
plt.subplot(231)
plt.xlabel('% Dev. of Recommended Song from Avg. User-Profile', size=6)
plt.ylabel('# of test samples')
plt.title('Song Duration', size=6)
plt.hist(avg_dur_array, bins=[10*i for i in range(0,20)], facecolor ="orange")

plt.subplot(232)
plt.xlabel('% Dev. of Recommended Song from Avg. User-Profile', size=6)
plt.ylabel('# of test samples')
plt.title('Song Key Signature', size=6)
plt.hist(avg_key_array, bins=[10*i for i in range(0,20)], facecolor ="green")

plt.subplot(233)
plt.xlabel('% Dev. of Recommended Song from Avg. User-Profile', size=6)
plt.ylabel('# of test samples')
plt.title('Song Tempo', size=6)
plt.hist(avg_tempo_array, bins=[10*i for i in range(0,20)], facecolor ="red")

plt.subplot(234)
plt.xlabel('% Dev. of Recommended Song from Avg. User-Profile', size=6)
plt.ylabel('# of test samples')
plt.title('Song Time Signature', size=6)
plt.hist(avg_time_array, bins=[10*i for i in range(0,20)], facecolor ="blue")

plt.subplot(235)
plt.title('Common artists heard by similar users', size=6)
plt.xlabel('% of Common Artists between user and most similar user', size=6)
plt.ylabel('# of test samples')
plt.hist(num_of_common_artists, facecolor ="purple")
plt.show()
