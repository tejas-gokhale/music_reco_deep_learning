import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

max_duration = 1819.76771
max_key_sig  = 11
max_tempo    = 262.828
max_time_sig = 7

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

def make_prediction(user_id, user_dict, user_song_dict, song_dict):
	(avg_duration, avg_key_sig, avg_tempo, avg_time_sig, artist_id_list) = make_user_profile(user_id, user_dict, song_dict)

	# Normalized 
	n_avg_duration = avg_duration/max_duration
	n_avg_key_sig  = avg_key_sig/max_key_sig
	n_avg_tempo    = avg_tempo/max_tempo
	n_avg_time_sig = avg_time_sig/max_time_sig

	min_distance = 0
	flag = False

	for song in song_dict:

		if (song not in user_song_dict[user_id]):

			artist_id 	 = str(song_dict[song][4])
			duration     = float(song_dict[song][10])/max_duration
			key_sig      = float(song_dict[song][11])/max_key_sig
			tempo        = float(song_dict[song][13])/max_tempo
			time_sig     = float(song_dict[song][14])/max_time_sig

			distance_square = ((duration - n_avg_duration)**2.0) + ((key_sig - n_avg_key_sig)**2.0) + ((tempo - n_avg_tempo)**2.0) + ((time_sig - n_avg_time_sig)**2.0)
			if (artist_id not in artist_id_list):
				distance_square += 1

			distance = math.sqrt(distance_square)


			if (flag == False):
				min_distance = distance
				recomm_song  = song
				flag = True

			elif (distance < min_distance):
				min_distance = distance
				recomm_song  = song

	return (recomm_song, min_distance)


def find_dev_stats(user_id, recomm_song, user_dict, user_song_dict, song_dict):

	(avg_duration, avg_key_sig, avg_tempo, avg_time_sig, artist_id_list) = make_user_profile(user_id, user_dict, song_dict)

	artist_id 	 = str  (song_dict[recomm_song][4])
	duration     = float(song_dict[recomm_song][10])
	key_sig      = float(song_dict[recomm_song][11])
	tempo        = float(song_dict[recomm_song][13])
	time_sig     = float(song_dict[recomm_song][14])
	
	if (artist_id in artist_id_list):
		artist_dev = 1
	else:
		artist_dev = 0


	duration_dev = 100.0*(abs(avg_duration - duration)/(avg_duration))
	key_sig_dev  = 100.0*(abs(avg_key_sig - key_sig)/(avg_key_sig + 0.00001))
	tempo_dev    = 100.0*(abs(avg_tempo - tempo)/(avg_tempo + 0.00001))
	time_sig_dev = 100.0*(abs(avg_time_sig - time_sig)/(avg_time_sig + 0.00001))

	dev_stats = [artist_dev, duration_dev, key_sig_dev, tempo_dev, time_sig_dev]

	return dev_stats

def main():


	song_dict = make_song_dict()
	user_dict, user_song_dict = make_user_dict()

	artist_dev_array   = []
	duration_dev_array = []
	key_sig_dev_array  = []
	tempo_dev_array    = []
	time_sig_dev_array = []

	for iteration in range(20000):

		user_id = random.choice(list(user_dict))

		(recomm_song , min_distance) = make_prediction(user_id, user_dict, user_song_dict, song_dict)
		dev_stats = find_dev_stats(user_id, recomm_song, user_dict, user_song_dict, song_dict)
		
		artist_dev_array.append(dev_stats[0])
		if (dev_stats[1] < 200):
			duration_dev_array.append(dev_stats[1])
		key_sig_dev_array.append(dev_stats[2])
		tempo_dev_array.append(dev_stats[3])
		time_sig_dev_array.append(dev_stats[4])

		if(iteration % 1000 == 0):
			print(iteration)	
	
	plt.subplot(231)
	plt.title('Duration Deviation Histogram', size=8)
	plt.hist(duration_dev_array, bins=[10*i for i in range(0,20)], facecolor ="orange")
	
	plt.subplot(232)
	plt.title('Key Signature Deviation Histogram', size=8)
	plt.hist(key_sig_dev_array, bins=[10*i for i in range(0,20)], facecolor ="green")
	
	plt.subplot(233)
	plt.title('Tempo Deviation Histogram', size=8)
	plt.hist(tempo_dev_array, bins=[10*i for i in range(0,20)], facecolor ="red")
	
	plt.subplot(234)
	plt.title('Time Signature Deviation Histogram', size=8)
	plt.hist(time_sig_dev_array, bins=[10*i for i in range(0,20)], facecolor ="blue")
	
	plt.subplot(235)
	plt.title('Previously Heard Artists Histogram', size=8)
	plt.hist(artist_dev_array, facecolor ="purple")
	plt.show()

main()