#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:31:02 2016

@author: dinara
"""
import pandas as pd
import numpy as np

# Read data from files 
train = pd.read_csv( '/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/trainDataVecs (smallerTrainSet).csv', header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( '/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/testDataVecs (smallerTestSet).csv', header=0, delimiter="\t", quoting=3 )


# Verify the number of reviews that were read (100,000 in total)
print ("Read %d labeled train docs\n" % (train["id"].size))
 
#trainDataVecs = np.loadtxt('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/trainDataVecs (smallerTrainSet)')
#testDataVecs = np.loadtxt('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/testDataVecs (smallerTestSet)')

#*******following are 3 ways of generating trainDataVecs and testDataVecs numpy arrays from csv files, all failed*****
#

#trainDataVecs = np.loadtxt(train, delimiter='\t', skiprows=5, usecols=["page_vector_200dim"] )
#testDataVecs = np.loadtxt(test, delimiter='\t', skiprows=5, usecols=["page_vector_200dim"] )

#trainData = pd.read_csv('/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/trainDataVecs (smallerTrainSet).csv', sep="\t", usecols=['page_vector_200dim'])
#testData = pd.read_csv('/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/testDataVecs (smallerTestSet).csv', sep="\t", usecols=['page_vector_200dim'])
#trainDataVecs = trainData.values
#testDataVecs = testData.values

#trainDataVecs = np.genfromtxt('/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/trainDataVecs (smallerTrainSet).csv', delimiter='\t', usecols=["page_vector_200dim"] )
#testDataVecs = np.genfromtxt('/home/dinara/word2vec/word2vec_gensim_ODP/csv_files/testDataVecs (smallerTestSet).csv', delimiter='\t', usecols=["page_vector_200dim"] )

#fit random forest
# Fit a random forest to the training data, using 100 trees

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 12, n_jobs = 10)

print ("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["catid"] )
#forest = forest.fit( trainDataVecs, train["id"] )
print ("Fitted, now predict...")


# Test & extract results 
result = forest.predict( testDataVecs )
print ("Predicted, now output...")

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "catid":result} )
#output.to_csv( "randomForest/Word2Vec_AverageVectors_smallTrainSet_12estimators_10jobs.csv", index=False, quoting=3 )