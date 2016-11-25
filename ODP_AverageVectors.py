#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:06:12 2016

@author: dinara
"""

import logging, gensim

    
if __name__ == '__main__':


    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    
    # Load model
    print ("Loading Word2Vec model...")
    model =  gensim.models.Word2Vec.load('kaggleODP_pages_200features_0minwords_10context')
    
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    #model.init_sims(replace=True)


    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")

  