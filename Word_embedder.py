# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:28:40 2018

@author: Arnaoty
"""
from gensim.models import Word2Vec

class Word_Embedder(object):

    def __init__(self, sentences, model = 'w2v', **params):
        self.embedding = model
        if self.embedding == 'w2v':
           self.word_model = Word2Vec(sentences, **params)
           
           
    def save_model(self, file):
        self.word_model.save(file)
        
    def load_model(self, file):
        self.word_model = Word2Vec.load(file)
        
    'Use this to start a new training or continue training on the current model'
    def train(self, sentences,  **params):
        if self.embedding == 'w2v':
            self.word_model.build_vocab(sentences) #, update = True)
            self.word_model.train(sentences, **params)
        
    def get_model(self):
        return self.word_model
    
    def get_vector(self,word):
        if self.embedding == 'w2v':
            return self.word_model.wv[word]