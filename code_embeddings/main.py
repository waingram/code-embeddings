# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:31:01 2018

@author: Arnaoty
"""

from rawCodeReader import Raw_Code_Reader
from Word_embedder import Word_Embedder
from Function_embedder import Function_Embedder
from nltk.corpus import PlaintextCorpusReader
from gensim.models.doc2vec import TaggedDocument
#from embedding_visualizer import tsne_plot
#from gensim.models import Doc2Vec


'''
This part of the code just generates embeddingds on the word level
'''
# batch_size = 42120 #10000
corpusDirectory = "E:\\PHD\\Thesis VT\\Data\\era_bcb_sample"
code_repository = Raw_Code_Reader(corpusDirectory)
programs_batch = code_repository.load_all_data()
# programs_batch = code_repository.load_next_batch(batch_size)
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


mdl2 = wrd_embd.get_model()
mdl2.wv.most_similar('--')
mdl2.wv.most_similar('count')
mdl2.wv.most_similar('delete')
mdl2.wv.most_similar('HttpGet')
mdl2.wv.most_similar('NullPointerException')
#tsne_plot(mdl)


'''
This part of the code generates embeddingds on the function level
'''
tkn_methods = code_repository.load_all_methods()
tkn_methods = code_repository.tokenize_loaded_methods()

'Need to give a name to each method in the training set. So, I name a method by its fileName + order_within_file'
corpusFiles = PlaintextCorpusReader(corpusDirectory, r'.*\.mthds')
labeled_files = zip(tkn_methods,corpusFiles.fileids())

labeled_methods = []
labeled_methods = [TaggedDocument(words=f, tags=[fid +"_"+ str(m)]) 
                                            for lf,fid in labeled_files for m,f in enumerate(lf)]

model = Function_Embedder(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.train(labeled_methods)
model.save_model('doc2v.mdl')

m = model.get_model()    
m.most_similar('2/sample/BinarySearch.mthds_1')
m.most_similar('10/sample/BubbleSort.mthds_1')
m.most_similar('2/sample/encryptFile.mthds_1')