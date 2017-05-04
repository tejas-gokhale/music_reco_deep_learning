import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

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
