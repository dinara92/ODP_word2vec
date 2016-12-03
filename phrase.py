#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:05:23 2016

@author: dinara
"""
import re

with open('../trainCategories.txt', 'r') as f:
    trainSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
# Set values for various parameters
num_features = 200    # Word vector dimensionality                      
min_word_count = 0   # Minimum word count                        
num_workers = 2       # Number of threads to run in parallel
context = 10         # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
import gensim

print ("Training model with phrase...")

bigram_transformer = gensim.models.Phrases(trainSentences)
model = word2vec.Word2Vec(bigram_transformer[trainSentences], workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model_name = "phrase_kaggleODP_categ_200features_0minwords_10context"
model.save(model_name)