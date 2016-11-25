#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:33:59 2016

@author: dinara
"""

import re

with open('trainPages.txt', 'r') as f:
    trainSentences = [re.sub("[^\w]", " ",  line).split() for line in f]

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 200    # Word vector dimensionality                      
min_word_count = 0   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10         # Context window size                                                                                    
#downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(trainSentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "kaggleODP_pages_200features_0minwords_10context"
model.save(model_name)