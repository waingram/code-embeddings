# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:30:29 2018

@author: Arnaoty
"""
from gensim.models import Doc2Vec

class Function_Embedder(object):
    
    def __init__(self, model = 'doc2v', **params):
        self.embedding = model
        if self.embedding == 'doc2v':
           self.fn_model = Doc2Vec(**params)
           
           
    def save_model(self, file):
        self.fn_model.save(file)
        
    def load_model(self, file):
        self.fn_model = Doc2Vec.load(file)
        
    'Use this to start a new training or continue training on the current model'
    def train(self, labeled_methods,  **params):
        if self.embedding == 'doc2v':
            self.fn_model.build_vocab(labeled_methods) #, update = True)
            for epoch in range(10):
                self.fn_model.train(labeled_methods,epochs = 2 , total_examples= self.fn_model.corpus_count)
                self.fn_model.alpha -= 0.002  # decrease the learning rate
                self.fn_model.min_alpha = self.fn_model.alpha  # fix the learning rate, no decay
        
    def get_model(self):
        return self.fn_model.docvecs
    
    def get_vector(self,word):
        if self.embedding == 'doc2v':
            return self.fn_model.wv[word]

    
    
