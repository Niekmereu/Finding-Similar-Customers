# Finding Similar Users
Minhashing, Local Sensitive Hashing

This algorithm finds similar users in the Yelp dataset review data: https://www.kaggle.com/yelp-dataset/yelp-dataset. It uses Minhashing, Local Sensitive Hashing and the Jaccard similarity.

The ratings itself are deleted. We subsequently only want to find users that have rated the same restaurants (similar users). 

# Minhashing 
The matrix is businesses x users with a 1 if a user rated a business. Implicit permutations (hash functions) are used to find the first row in which a user has a 1. The matrix that is formed is called a signature matrix. This signature matrix is much smaller than the original one. 

# Local Sensitive Hashing
The signature matrix is then divided in b bands with r rows per band. For each band, the parts of the columns of the signature matrix that belong to the band are hashed to a hash table with k buckets. We speak of a candidate pair if users are hashed to the same bucket and the similarity of their signatures is larger than crit1. Finally we check the true similarity of all candidate pairs, which should be higher than crit2. 


# Parameters

- n_naive: how many times must a customer must appear in the training set before we apply matrix factorization for this specfic customer.    It is generally a good idea to leave the customers that appear only once (n_naive = 2). 
- crit1: critical similarity of the signature to become candidates
- crit2: how similar must two users be with regard to their original date in order to be called similar
 
