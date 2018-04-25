# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:29:38 2018

@author: Arnaoty
"""

from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import PlaintextCorpusReader
import random

java_model_file = "doc2vec.mdl"
python_model_file = ""
out_java_model = "j_fun_vectors.mdl"
out_python_model = "p_fun_vectors.mdl"

java_model =  Doc2Vec.load(java_model_file)
python_model =  Doc2Vec.load(python_model_file)


'''
Constructing model files for the different languages
'''


java_pairs = [(x,java_model.docvecs[x]) for x in java_model.docvecs.doctags.keys()]
python_pairs = [(x,python_model.docvecs[x]) for x in python_model.docvecs.doctags.keys()]


def write_model_to_file(model_pair,fname):
    f = open(fname,'w')
    for x,v in model_pair:
        f.write(x.strip())
        f.write(' ')
        f.write(' '.join([ str(s).strip() for s in v]))
        f.write('\n')
    f.close()
    
write_model_to_file(java_pairs,out_java_model)
write_model_to_file(python_pairs,out_python_model)


'''
Constructing sample alignment seeds.

steps:
    load fnames from one language
    choose a random sample
    move to corresponding folder
    choose one file, mark as selected for removing redunduncy
    print the pair fn1 ||| fn2
'''

#java_files_path = "E://PHD//Thesis VT//SourceCodeEmbedding//code-embeddings//test_data//Java"
#python_files_path = "E://PHD//Thesis VT//SourceCodeEmbedding//code-embeddings//test_data//python"

java_files_path = "..//test_data//Java"
python_files_path = "..//test_data//python"
alignment_file = 'align-sample.txt'

java_corpus = PlaintextCorpusReader(java_files_path, r'.*\.java')
python_corpus = PlaintextCorpusReader(python_files_path, r'.*\.py')

java_files = java_corpus.fileids()
python_files = python_corpus.fileids()

randomset = random.sample(java_files,150)
sampled = []
pairs = []

for s in randomset:
#    print(s)
    selected = ''
    py_lst = [x for x in python_files if s.split('/')[0] in x]
    selected = random.choice(py_lst)
    i=0
    while i<10 and selected in sampled:
        selected = random.choice(py_lst)
        i += 1
    if i == 10:
        continue
    sampled += [selected]
#    print(sampled)
    pairs += [s + ' ||| ' + selected]
    
with open(alignment_file,'w') as f:
    f.write("\n".join(pairs))


