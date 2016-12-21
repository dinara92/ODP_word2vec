#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:49:16 2016

@author: dinara
"""

import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
from gensim.models import word2vec
 
 
def main():
 
    #save .bin model embeddings to .txt
    #model = word2vec.Word2Vec.load_word2vec_format('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    #model.save_word2vec_format('/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/GoogleNews-vectors-negative300.txt', binary=False)

    #embeddings_file = sys.argv[1]
    embeddings_file = '/home/dinara/word2vec/word2vec_gensim_ODP/ODP_word2vec/GoogleNews-vectors-negative300.txt'
    wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000,:])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
 
def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary
 
if __name__ == '__main__':
    main()
    