# -*- coding: utf-8 -*-

import numpy as np
import operator
import time

users = 6040
items = 3952



def data_reader_cv () :
    rating = "./ratings.dat"
    with open(rating, 'r') as data :
        ratings = data.readlines()  ## rating is list userid :: movieid :: rating :: timestamp
    dataset = []
    for line in ratings :
        sp = line.split('::') ##parsing lines into [userid, movieid,rating, timestamp]
        dataset.append([int(sp[0]),int(sp[1]),int(sp[2])])
    return dataset  
    
def uim_builder (train_set) :
    user_item_matrix = np.zeros((6040,3952))
    for rows in train_set :
        user_item_matrix[rows[0]-1][rows[1]-1] = rows[2]
    return user_item_matrix

def isolate_user2 (i, j, uim) : ##list
    vector_i = []
    vector_j = []
    for rows in uim :
        if((rows[i-1] != 0) & (rows[j-1] != 0)) :
            vector_i.append(rows[i-1])
            vector_j.append(rows[j-1])   
    return [np.asarray(vector_i),np.asarray(vector_j)]

from numpy import dot
from numpy.linalg import norm

def sim_cos (v) :
    if v[0].size != 0 : 
        return dot(v[0], v[1])/(norm(v[0])*norm(v[1]))
    return 0

def cos_matrix_builder (uim) :
    
    cos_matrix = np.ones((items, items))
    for i in range(0, items, 1) :
        for j in range(0, items, 1) :
            if i == j :
                break
            S_ij = sim_cos(isolate_user2(i,j, uim))
            np.around(S_ij, decimals=5)
            if (i % 100 == 0) & (j == 0) :
                print("S_{0:d},{1:d} = {2:f}".format(i,j, S_ij))
            cos_matrix[i][j] = S_ij
            cos_matrix[j][i] = S_ij

    return cos_matrix

def k_similar_item (cos_matrix) : ##
    start_time = time.time()
    k_matrix_list = []
    for k in range(25, 201, 25) : ## modei size가 25씩 증가    
        dicList = []
        for row in cos_matrix : ## cos_matrix의 가로 
            num = 0
            min_key = 0
            dic = {} ## key : value = index : similarity
            for index in range(0,items,1):
                if row[index] == 0 :
                    continue
                if num == 0 :
                    dic[index] = row[index]
                    min_key = index
                    num+=1
                elif num < k :
                    dic[index] = row[index]
                    num+=1
                    if dic[min_key] > row[index]:
                        min_key = index
                elif num == k :
                    if row[index] < dic[min_key]:
                        continue
                    if row[index] > dic[min_key]:
                        del(dic[min_key])
                        dic[index] = row[index]
                        sortdic = sorted(dic.items(), key=operator.itemgetter(1))
                        min_key = sortdic[0][0]
            dic_keys = list(dic.keys())
            if len(dic_keys) < k:
                while( len(dic_keys) < k) : ## not sure
                    dic_keys.append(-1)                    
            dicList.append(dic_keys)
        most_similar_items = np.array(dicList)
        k_matrix_list.append(most_similar_items)
    print("---k_similar_matrix builder costs  %s seconds ---" % (time.time() - start_time))
    return k_matrix_list 


def prediction (u,i, uim, cos_matrix, k_similar_matrix) :
    sR = 0
    absoluteS = 0
    for similar_items in k_similar_matrix[i-1] :
        if similar_items == -1 :
            break
        sR = sR + uim[u-1][similar_items-1]*cos_matrix[i-1][similar_items-1]
        absoluteS = absoluteS + abs(cos_matrix[i-1][similar_items-1])
    return np.around(sR/absoluteS, decimals = 5)
    
def MAE (test_set) : ##test_set has same shape with uim userid::movieid::rating
    sum = 0
    for rows in test_set :
        sum = abs(prediction(rows[0][1]) - rows[2])
    return sum/len(test_set)

'''
def main ():
    cos_matrix = cos_matrix_builder()
    np.savetxt('./cos_sim_matrix', cos_matrix)
    
if __name__ == '__ main()__' :
    main()
''' 