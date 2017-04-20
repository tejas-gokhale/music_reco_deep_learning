## Recommender Systems
# Author:   Tejas Gokhale
# Date:     20-MAR-2017

## imports
import os
import numpy as np
import time
import operator
import math

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
    
    # print(len(sim))
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

my_user = '8305c896f42308824da7d4386f4b9ee584281412'
makeRecommendation(my_user)

            
        