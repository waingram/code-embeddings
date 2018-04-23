# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:51:11 2018

@author: Arnaoty
"""

'''
This one needs python 2.7
'''

#from lda2vec import preprocess, Corpus


model = LDA2Vec(n_words, max_length, n_hidden, counts)
model.add_component(n_docs, n_topics, name='document id')
model.fit(clean, components=[doc_ids])

topics = model.prepare_topics('document_id', vocab)
prepared = pyLDAvis.prepare(topics)
pyLDAvis.display(prepared)