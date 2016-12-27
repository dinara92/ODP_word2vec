#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:00:31 2016

@author: dinara
"""
import re, random
# random
from random import shuffle
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

class LabeledLineSentenceOriginal(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        with open(self.filename, 'r') as f:
            for uid, line in enumerate(f):
                yield LabeledSentence(re.sub("[^\w]", " ", line).split(), ['SENT_%s' % uid])
    #def sentences_perm(self):
        #shuffle(self.sentences)
        #return self.sentences
        
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(re.sub("[^\w]", " ",  line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(re.sub("[^\w]", " ",  line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

#with open('/home/dinara/word2vec/word2vec_gensim_ODP/text_files_for_training/trainPages.txt', 'r') as f:
    #trainSentences = [re.sub("[^\w]", " ",  line).split() for line in f]
    #print(trainSentences)


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


sources = {'/home/dinara/word2vec/word2vec_gensim_ODP/text_files_for_training/trainPages.txt':'TRAINING'}
source = '/home/dinara/word2vec/word2vec_gensim_ODP/text_files_for_training/trainPages.txt'

sentences = LabeledLineSentenceOriginal(source)
                    
model = Doc2Vec(min_count=5, window=8, size=100, sample=1e-4, negative=5, workers=4)  # use fixed learning rate
model.build_vocab(sentences)


for epoch in range(10):
    #model.train(sentences.sentences_perm())
    model.train(sentences)


model.save('train_pages_ODP_doc2vec')
#model = Doc2Vec.load('example.model')  
