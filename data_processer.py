# -*- coding: utf-8 -*-

import numpy as np
def data_reader () : #read rating file
    rating = "C:/dataset/ml-1m/ratings.dat"
    with open(rating, 'r') as data :
        ratings = data.readlines()
    return ratings ## rating is list userid :: movieid :: rating :: timestamp

def data_processor (ratings) : #build user_item_matrix
    user_item_matrix = np.zeros((6040,3952))
    for line in ratings :
        sp = line.split('::') ##parsing lines into [userid, movieid,rating, timestamp]
        user_item_matrix[int(sp[0])-1][int(sp[1])-1] = int(sp[2])
    return user_item_matrix