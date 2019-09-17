# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:11:45 2019

@author: Niek
"""

import numpy as np
import itertools
from scipy.sparse import csc_matrix
import pandas as pd

#Function to create a matrix of hash functions. All randomness is here.
def scramble(a, axis=-1):
    np.random.seed(1234)
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled


#data, how many times must user appear, crit for signatures similarity, crit for true similarity
def main(d, N, crit1, crit2):

    #Find the indices of items that appear less than N times in the training data

    unique, counts = np.unique(d['col0'], return_counts=True)
    unique = dict(zip(unique, counts))
    index = [k for k,v in unique.items() if int(v) in range(1, N) ]

    #find indexing
    d = d[~d['col0'].isin(index)]
    d = d.sort_values('col0')
    d['col0'] = pd.factorize(d.col0)[0] + 1
    d = d.sort_values('col1')
    d['col1'] = pd.factorize(d.col1)[0] + 1
    d = d.sort_values('col0')

    #bands and rows per band
    n_hash = 5 * 20
    bands = 20
    r = int(n_hash / bands)
    
    #Initialize the signatures matrix
    signatures = np.array(np.ones((n_hash, np.max(np.array(d)[:,0]))) * np.inf)
    
    #Convert to pandas because for handy aggregate function, don't work in pandas
    #because it's very slow
    temp = d.groupby('col0').count().cumsum()
    temp = np.array(temp)
    
    #Back to np, seems to be faster
    d = np.array(d)
    #Find max number of businesses and users
    n_biss = np.max(d[:,1])
    n_users = np.max(d[:,0])
    
    #Create the hash matrix
    n_h = scramble(np.tile(range(n_biss), (n_hash, 1)), axis = 1).T  

    #Format the input data to something workable
    in_matrix = csc_matrix((n_biss + 1, n_users+1))

    in_matrix[d[:,1] , d[:,0] ] = 1

    #Signature matrix
    print("Consolidation of the signature matrix is starting.")
    for i in range(0, np.max(d[:,0]-1)):

       m = d[temp[i][0]:temp[i + 1][0],1] 
       temp3 = n_h[m - 1, :] 
       temp4 = np.amin(temp3, axis = 0)  
       signatures[:, i] = temp4
           
    #Initializa out as np array with two zeroes, remove later
    out = np.zeros(shape = (1, 2))
    #idx contains first row of each band
    idx = list(range(0, n_hash, r))
    for i in range(bands):
        print("Working on band number:", i )
        bucket_dict = dict()
        ID = np.vstack((signatures[idx[i]:idx[i]+r,], np.arange(0, n_users)))
        for j in ID.T:
            if tuple(j[0:r]) in bucket_dict:
                bucket_dict[tuple(j[0:r])].append(int(j[r]))
            else:
                bucket_dict[tuple(j[0:r])] = list()
                bucket_dict[tuple(j[0:r])].append(int(j[r]))
        
        #Remove all keys where length of values is smaller than 2
        bucket_dict = {k:v for k,v in bucket_dict.items() if len(v) > 1}

        print("The number of buckets is:", len(bucket_dict))

        #Now create a list containing all combinations of possible candidates
        l = []
        for i in list(bucket_dict.values()):
            if len(i) < 3:
                l.append(i)
            if len(i) > 2:
                temp = list(itertools.combinations(i, 2))
                for j in temp:
                    l.append(j)
        print("The total number of combinations to be checked in signature matrix is:",
              len(l))
                 
        #For all combinations, calculate sim based on signature matrix
        sim = np.zeros(shape = (len(l), 1))
        c = 0
        for v in l:
            sim[c,0] = np.sum(signatures[:,v[0]] == signatures[:,v[1]], axis = 0)/n_hash
            c = c + 1
       
        #Candidates if sim is higher than crit
        candidates = np.array(l)[(np.where(sim >= crit1)[0]), :]
        candidates = candidates + 1

        print("The total number of candidates is:", len(candidates))
        
        #Check the true similiarity, append if larger than crit2
        simmie_real = np.zeros(shape = (len(candidates), 1))
        c = 0
        for v in candidates:
            
            ic = np.intersect1d(in_matrix[:,v[0]].indices, 
                                in_matrix[:,v[1]].indices)
            a = in_matrix[:,v[0]].nnz
            b = in_matrix[:,v[1]].nnz
            simmie_real[c,0] = (len(ic)/(a + b - len(ic)))
            c = c + 1
      

        out_band = candidates[np.where(simmie_real >= crit2),:][0]
        
        out = np.vstack((out, out_band))
        
    out = out.astype(int)
    out = np.delete(out, 0, axis = 0)
    out = np.sort(out)
    out = np.unique(out, axis=0)
    
    #Now we can also remove the accidental inverse duplicates (not sure if unique does that)
    np.delete(out, np.where(out[:,1] < out[:,0]))

    
    print("")
    print("The total number of pairs is:", len(out))
    
    np.set_printoptions(suppress=True)
    np.savetxt("results.txt", out, fmt='%i')
       
    return(out)
    
#The function with seed and loation
ratings2 = np.genfromtxt("Review.csv", usecols=(1, 2, 3), delimiter=';', dtype='int')
ratings2 = np.delete(ratings2, 0, axis =0)
ratings2 = np.delete(ratings2, 2, axis =1)
ratings2 = pd.DataFrame(ratings2,
         columns = ["col0", "col1"])      

#Run algortrithm 
main(ratings2, 2, 0.4, 0.2)

