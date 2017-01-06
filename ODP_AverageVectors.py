#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:06:12 2016

@author: dinara
"""

import logging, gensim, re
import numpy as np
import pandas as pd
import sklearn,csv

num_features = 1000

with open('/Users/dinaraDILab/java_projects/ODP-Word2Vec/trainPages.txt', 'r') as f:
    trainSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

with open('/Users/dinaraDILab/java_projects/ODP-Word2Vec/testPages.txt', 'r') as f:
    testSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

# ****** Define functions to create average word vectors
#

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        print("\tFit")       
        tfidf = sklearn.TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = sklearn.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        print("\tTransform")       
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
                
def sent_vectorizer(sent, model, num_features, index2word_set):
    sent_vec = np.zeros(num_features)
    numw = 0
    
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    #index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total

    for w in sent:
        if w not in index2word_set:
            continue
        #try:
        sent_vec = np.add(sent_vec, model[w])
        numw+=1
        #except:
            #pass
    #return sent_vec / np.sqrt(sent_vec.dot(sent_vec)) /numw
    #sent_vec = np.divide(sent_vec,np.sqrt(sent_vec.dot(sent_vec)))
    sent_vec = np.divide(sent_vec, numw)
    return sent_vec

def getAvgFeatureVecs(sentences, model, num_features, index2word_set):
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
           num_features, index2word_set)
       #

       # Increment the counter
       counter = counter + 1.
    return featureVecs

def vect_adder(sent, model1, model2, num_features):
    sent_vec = np.zeros(num_features)
    numw = 0
    for w in sent:
        try:
            #sent_vec = np.add(sent_vec, model1[w], model2[w]) 
            sent_vec = np.add(sent_vec, model1[w]) 
            sent_vec = np.add(sent_vec, model2[w])
            numw+=1
        except:
            pass
    #return sent_vec / np.sqrt(sent_vec.dot(sent_vec)) /numw
    #sent_vec = np.divide(sent_vec,np.sqrt(sent_vec.dot(sent_vec)))
    sent_vec = np.divide(sent_vec, numw)
    sent_vec = np.divide(sent_vec, 2)
    return sent_vec

def get2ModelsAvgFeatureVecs(sentences, trainVecs1, trainVecs2):
    counter = 0
    featureVecsAve = np.zeros((len(sentences),num_features),dtype="float32")
    for sent in sentences:
        # Print a status message every 1000th review
        if counter%1000. == 0.:
           print ("Document %d of %d" % (counter, len(sentences)))
           
        #featureVecsAve[counter] = np.add (trainVecs1[counter], trainVecs2[counter])
        #featureVecsAve[counter] = np.divide(featureVecsAve[counter], 2)
        featureVecsAve[counter] = vect_adder(sent, trainVecs1, trainVecs2, num_features)
        
        # Increment the counter
        counter = counter + 1.
        
    return featureVecsAve
    
if __name__ == '__main__':

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    
    # Load model
    print ("Loading Word2Vec model...")
    #model =  gensim.models.Word2Vec.load('/home/dinara/word2vec/word2vec_gensim_ODP/trained_models_pages/train_pages_10context/kaggleODP_pages_200features_0minwords_10context')
    #model = gensim.models.Word2Vec.load_word2vec_format('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
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
    #print ("\tCreating average feature vecs for train docs")
    #trainDataVecs = getAvgFeatureVecs( trainSentences, model, num_features )
    #maybe, better to save (all ODP pages - testpages) as trainDataVecs
    #np.savetxt('trainDataVecs_phrase_bigram_200f_0minw_10context', trainDataVecs)
    #print ("\tCreating average feature vecs for test docs")
    #testDataVecs = getAvgFeatureVecs( testSentences, model, num_features )
    #np.savetxt('testDataVecs_phrase_bigram_200f_0minw_10context', testDataVecs)

    #map key(word) -> to value (200-dim vector)
    #w2v = dict(zip(model.index2word, model.syn0))
    #print("This is dictionary")

    #TfidfEmbeddingVectorizer(w2v)
    #with open('dict.csv', 'w') as csv_file:
    #    writer = csv.writer(csv_file)
    #    for key, value in w2v.items():
    #       writer.writerow([key, value])
    
    #tfidf = gensim.models.tfidfmodel.TfidfModel(trainSentences)
    #tfidf.save('/tmp/trainPages.tfidf_model')
    
    #model1 =  gensim.models.Word2Vec.load('/home/dinara/word2vec/word2vec_gensim_ODP/trained_models_pages/train_pages_10context_300f/kaggleODP_pages_300features_0minwords_10context')
    #model2 = gensim.models.Word2Vec.load_word2vec_format('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
    model = gensim.models.Word2Vec.load('/Users/dinaraDILab/word2vec/en_wiki/en_1000_no_stem/en.model')
    #model = gensim.models.Word2Vec.load('/Users/dinaraDILab/word2vec/word2vec_py/ODP_word2vec/allODPpages_300features_0minwords_10context')
    index2word_set = set(model.index2word)

    print ("\tCreating average feature vecs for train docs")
    trainDataVecs = getAvgFeatureVecs( trainSentences, model, num_features, index2word_set )
    np.savetxt('/Users/dinaraDILab/word2vec/word2vec_py/ODP_word2vec/new_en_wiki/trainDocs_en_wiki', trainDataVecs)
    print ("\tCreating average feature vecs for test docs")
    testDataVecs = getAvgFeatureVecs( testSentences, model, num_features, index2word_set )
    np.savetxt('/Users/dinaraDILab/word2vec/word2vec_py/ODP_word2vec/new_en_wiki/testDocs_en_wiki', testDataVecs)

    
    #print ("\tCreating average feature vecs for train docs of Model1 and Model2")
    #dataVecs = get2ModelsAvgFeatureVecs(testSentences, model1, model2)
    #np.savetxt('get2ModelsAvgFeatureVecs_test_2', dataVecs)