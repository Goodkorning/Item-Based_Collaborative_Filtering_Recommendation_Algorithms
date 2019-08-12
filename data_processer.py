# -*- coding: utf-8 -*-

import numpy as np
import operator
import time

users = 6040
items = 3952

def data_reader () : #read rating file
    rating = "C:/dataset/ml-1m/ratings.dat"
    with open(rating, 'r') as data :
        ratings = data.readlines()  ## rating is list userid :: movieid :: rating :: timestamp
    user_item_matrix = np.zeros((6040,3952))
    for line in ratings :
        sp = line.split('::') ##parsing lines into [userid, movieid,rating, timestamp]
        user_item_matrix[int(sp[0])-1][int(sp[1])-1] = int(sp[2])
    return user_item_matrix

uim = data_reader()

def isolate_user2 (i, j) : ##list
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

def cos_matrix_builder () :
    
    cos_matrix = np.ones((items, items))
    for i in range(0, items, 1) :
        for j in range(0, items, 1) :
            if i == j :
                break
            S_ij = sim_cos(isolate_user2(i,j))
            if (i % 100 == 0) & (j == 0) :
                print("S_{0:d},{1:d} = {2:f}".format(i,j, S_ij))
            cos_matrix[i][j] = S_ij
            cos_matrix[j][i] = S_ij

    return cos_matrix

def k_similar_item (matrix) : ##
    start_time = time.time()
    k_matrix_list = []
    for k in range(25, 201, 25) : ## modei size가 25씩 증가    
        dicList = []
        for row in matrix : ## cos_matrix의 가로 
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
    


def main ():
    cos_matrix = cos_matrix_builder()
    np.savetxt('./cos_sim_matrix', cos_matrix)
    
if __name__ == '__ main()__' :
    main()
    