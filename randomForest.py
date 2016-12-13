#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:31:02 2016

@author: dinara
"""
import pandas as pd
import numpy as np

# Read data from files 
train = pd.read_csv( '/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/trainDataVecs.csv', header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( '/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/testDataVecs (copy).csv', header=0, delimiter="\t", quoting=3 )


# Verify the number of reviews that were read (100,000 in total)
print ("Read %d labeled train docs\n" % (train["id"].size))
 
trainDataVecs = np.loadtxt('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/trainDataVecs')
testDataVecs = np.loadtxt('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/testDataVecs')

#fit random forest
# Fit a random forest to the training data, using 100 trees

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 2, n_jobs = 4)

print ("Fitting a random forest to labeled training data...")
#forest = forest.fit( trainDataVecs, train["catid"] )
forest = forest.fit( trainDataVecs, train["id"] )
print ("Fitted, now predict...")


# Test & extract results 
result = forest.predict( testDataVecs )
print ("Predicted, now output...")

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "catid":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )