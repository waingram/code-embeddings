# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:31:01 2018

@author: Arnaoty
"""

from rawCodeReader import Raw_Code_Reader
from Word_embedder import Word_Embedder
#from embedding_visualizer import tsne_plot

batch_size = 42120 #10000
corpusDirectory = "E:\\PHD\\Thesis VT\\Data\\era_bcb_sample"
code_repository = Raw_Code_Reader(corpusDirectory)
programs_batch = code_repository.load_next_batch(batch_size)
tokenized_code = code_repository.tokenize_loaded_data()
print('Data Loaded')
print('Start training')
wrd_embd = Word_Embedder(tokenized_code,'w2v', size=100, window=5, min_count=1, iter = 5, workers=4)
wrd_embd.save_model('word2v.mdl')

#curr = batch_size
#while code_repository.has_next():
#    print('training on programs: ' + str(curr) + " to " + str(curr + batch_size))
#    curr += batch_size
#    programs_batch = code_repository.load_next_batch(batch_size)
#    tokenized_code = code_repository.tokenize_loaded_data()
#    wrd_embd.load_model('word2v.mdl')
#    wrd_embd.train(tokenized_code)
#    wrd_embd.save_model('word2v.mdl')
#    
print()
print("Training finished")


#mdl = wrd_embd.get_model()
#tsne_plot(mdl)
