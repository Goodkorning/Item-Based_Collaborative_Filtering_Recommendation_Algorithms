# -*- coding: utf-8 -*-

import numpy as np

def data_reader () : #read rating file
    rating = "C:/dataset/ml-1m/ratings.dat"
    with open(rating, 'r') as data :
        ratings = data.readlines()  ## rating is list userid :: movieid :: rating :: timestamp
    user_item_matrix = np.zeros((6040,3952))
    for line in ratings :
        sp = line.split('::') ##parsing lines into [userid, movieid,rating, timestamp]
        user_item_matrix[int(sp[0])-1][int(sp[1])-1] = int(sp[2])
    return user_item_matrix

def isolate_user2 (i, j) : ##list
    uim = data_reader()
    vector_i = []
    vector_j = []
    for rows in uim :
        if((rows[i-1] != 0) & (rows[j-1] != 0)) :
            vector_i.append(rows[i-1])
            vector_j.append(rows[j-1])