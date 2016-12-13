#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:06:12 2016

@author: dinara
"""

import logging, gensim, re
import numpy as np
import pandas as pd

num_features = 200

with open('../trainPages.txt', 'r') as f:
    trainSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

with open('../testPages.txt', 'r') as f:
    testSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

# ****** Define functions to create average word vectors
#

def sent_vectorizer(sent, model, num_features):
    sent_vec = np.zeros(num_features)
    numw = 0
    for w in sent:
        try:
            sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    #return sent_vec / np.sqrt(sent_vec.dot(sent_vec)) /numw
    sent_vec = np.divide(sent_vec,np.sqrt(sent_vec.dot(sent_vec)))
    sent_vec = np.divide(sent_vec, numw)
    return sent_vec

def getAvgFeatureVecs(sentences, model, num_features):
    # Given a set of sentences (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    featureVecs = np.zeros((len(sentences),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for sent in sentences:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Document %d of %d" % (counter, len(sentences)))
       # 
       # Call the function (defined above) that makes average feature vectors
       featureVecs[counter] = sent_vectorizer(sent, model, \
           num_features)
       #

       # Increment the counter
       counter = counter + 1.
    return featureVecs
    
if __name__ == '__main__':

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    
    # Load model
    print ("Loading Word2Vec model...")
    model =  gensim.models.Word2Vec.load('../train_pages_10context/kaggleODP_pages_200features_0minwords_10context')
    
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    #model.init_sims(replace=True)
    
    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.
    
    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above
    print ("\tCreating average feature vecs for train docs")
    trainDataVecs = getAvgFeatureVecs( trainSentences, model, num_features )
    #maybe, better to save (all ODP pages - testpages) as trainDataVecs
    #np.savetxt('trainDataVecs', trainDataVecs)
    print ("\tCreating average feature vecs for test docs")
    testDataVecs = getAvgFeatureVecs( testSentences, model, num_features )
    #np.savetxt('testDataVecs', testDataVecs)

    