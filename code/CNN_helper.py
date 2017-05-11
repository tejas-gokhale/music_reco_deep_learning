import csv
import math
import random
import operator
import numpy as np
import matplotlib.pyplot as plt
import os

max_duration = 1819.76771
max_key_sig  = 11
max_tempo    = 262.828
max_time_sig = 7



def make_user_dict():
	user_dict = dict()

	with open('pruned_triplets.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			user_id = row[0]
			song_id = row[1]
			plays   = int(row[2])

			if (user_id in user_dict):
				val = user_dict[user_id]
				val = val + [(song_id, plays)]
				user_dict[user_id] = val
			else:
				user_dict[user_id] = [(song_id, plays)]

	return user_dict


def make_song_dict():

	song_dict = dict()
	with open('formatted_song_csv.csv', newline='') as csvfile:
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



def make_song_to_user_dict():

	song_to_user_dict = dict()

	with open('pruned_triplets.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			user_id = row[0]
			song_id = row[1]
			plays   = row[2]

			if (song_id in song_to_user_dict):
				val = song_to_user_dict[song_id]
				val = val + [user_id]
				song_to_user_dict[song_id] = val
			else:
				song_to_user_dict[song_id] = [user_id]

	return song_to_user_dict


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

	return (avg_duration, avg_key_sig, avg_tempo, avg_time_sig)


def make_song_to_user_profile_dict(song_to_user_dict, user_dict, song_dict):
	song_to_user_profile_dict = dict()

	for song_id in song_to_user_dict:
		for user_id in song_to_user_dict[song_id]:
			(avg_duration, avg_key_sig, avg_tempo, avg_time_sig) = make_user_profile(user_id, user_dict, song_dict)

			for (song, plays) in user_dict[user_id]:
				if (song == song_id):
					rating = plays

			if (song_id in song_to_user_profile_dict):
				song_to_user_profile_dict[song_id] += [[user_id, avg_duration, avg_key_sig, avg_tempo, avg_time_sig, rating]]
			else:
				song_to_user_profile_dict[song_id] = [[user_id, avg_duration, avg_key_sig, avg_tempo, avg_time_sig, rating]]

	return song_to_user_profile_dict




def cnn_helper():

	user_dict                 = make_user_dict()

	"""user_dict is a dictionary which uses user_id as the key.
	   The value is of the form [(song1, #plays), (song2, #plays).....]
	"""
	song_dict                 = make_song_dict()
	
	"""song_dict is a dictionary which uses song_id as the key.
	   The value is of the form [song_no, song_id, album_id, album_name, artist_id, artist_latitude, 
	   							 artist_location, artist_longitude, artist_name, danceability, duration, 
	   							 key_signature, key_signature_conf, tempo, time_signature, time_signature_conf,
	   							 title, year]

	"""
	song_to_user_dict         = make_song_to_user_dict()

	"""song_to_user dict is a dictionary which uses song_id as the key. 
	   The value is of the form [user_id1, user_id2....]
	"""

	song_to_user_profile_dict = make_song_to_user_profile_dict(song_to_user_dict, user_dict, song_dict)

	""""song_to_user_profile_dict is a dictionary which uses song_id as the key
		The value is of the form [[user_id1, avg_duration_u1, avg_key_sig_u1, avg_tempo_u1, avg_time_sig_u1, rating_u1],
								  [user_id2, avg_duration_u2, avg_key_sig_u2, avg_tempo_u2, avg_time_sig_u2, rating_u2],
								  .
								  .
								  																			]
	"""
	return song_to_user_profile_dict







with open('../data/mxm_dataset_train.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    data_np = np.array(data)

corr_tuples = []
with open('../data/unique_tracks.txt', 'r', encoding='utf-8') as f:
    song_track_corr = f.readlines()
    for line in song_track_corr:
        keyval = line.split('<SEP>')
        key = keyval[0]
        val = keyval[1]
        corr_tuples.append((key, val))
corr_dict = dict(corr_tuples)



def loadMSD():
	datapath = os.path.join('../data/', 'triplets.txt')
	with open(datapath, 'r', encoding="utf8") as f:
		datalines = f.readlines()
	prefs = {}
	for line in datalines:
		(uid, mid, rating) = line.split('\t')
		prefs.setdefault(uid, {})
		prefs[uid][mid] = float(rating)
	return prefs

        
def pearsonSimilarity(u_a, u_b, user_data):

    ratings_a = user_data[u_a]
    ratings_b = user_data[u_b]
    
    avg_ratings_a = sum(ratings_a.values())/len(ratings_a)
    avg_ratings_b = sum(ratings_b.values())/len(ratings_b)
    
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


def makeRecommendation(my_user, user_dict, user_data):

	songs_my = [k for k,v in user_data[my_user].items()]
	sim = dict()
	for user in user_dict:
		set_my, set_user, sim[user] = pearsonSimilarity(my_user, user, user_data)

	top10 = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)[:10]
	return top10

####################
num_keys = 5001
####################


"""
Use to generate input vector to deep net given a user. 
Taking a user makes input vector for songs of top10 similar users 
"""


def generate_input_vector(my_user):
	user_data = loadMSD()
	user_dict = make_user_dict()
	song_dict = make_song_dict()
	top10 = makeRecommendation(my_user, user_dict, user_data)

	for i in range(len(data)):
		data[i] = data[i].strip().split(',')
	
	input_vector = []

	for entry in top10: #going through top10 users
		user = entry[0]

		(avg_duration, avg_key_sig, avg_tempo, avg_time_sig) = make_user_profile(user, user_dict, song_dict)

		for user_entry in user_dict[user]: #Going through all songs for the user
			song = user_entry[0]

			vector_row = []
			for i in range(len(data)): #finding song BOW
				if (song == corr_dict[data[i][0]]):

					inp = [0]*num_keys
					inp[0] = song

					for j in range(2, len(data[i])):
						key = int(data[i][j].split(':', 1)[0])
						val = int(data[i][j].split(':', 1)[1])
						inp[key] = val

					vector_row = inp + [avg_duration, avg_key_sig, avg_tempo, avg_time_sig]
					input_vector.append(vector_row)
					#print(song)
					break
	return input_vector


"""
my_user = '8305c896f42308824da7d4386f4b9ee584281412'
input_vector = generate_input_vector(my_user)

with open('../data/recomm_test.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for row_vector in input_vector:
    	spamwriter.writerow(row_vector)
"""