#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:58:52 2016

@author: dinara
"""

from sklearn.cluster import KMeans
import time, gensim, numpy as np

start = time.time() # Start time

 # Load model
print ("Loading Word2Vec model...")
model =  gensim.models.Word2Vec.load('../train_pages_10context/kaggleODP_pages_200features_0minwords_10context')

#trainDataVecs = np.loadtxt('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/trainDataVecs')

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
#can do same by substituting word_vectors to doc_vectors
#doc_vectors = trainDataVecs
num_clusters = word_vectors.shape[0] // 5


# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.index2word, idx ))

# For the first 10 clusters
for cluster in range(0,10):
    #
    # Print the cluster number  
    print ("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print (words)