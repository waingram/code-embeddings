# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:29:38 2018

@author: Arnaoty
"""

from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import PlaintextCorpusReader
from gensim.models.doc2vec import TaggedDocument
import random
import io
import tokenize

java_model_file = "..//models//github-java-vectors.bin"
python_model_file = "..//models//github-python-vectors.bin"
out_java_model = "..//models//j_fun_vectors.mdl"
out_python_model = "..//models//p_fun_vectors.mdl"

java_model =  Doc2Vec.load(java_model_file)
python_model =  Doc2Vec.load(python_model_file)

#
#jrosetta_keys = [k for k in java_model.docvecs.doctags.keys() if 'rosetta' in k]
#prosetta_keys = [k for k in python_model.docvecs.doctags.keys() if 'rosetta' in k]
#
#def form_pair(j,p):
#    candidates_j = [k for k in jrosetta_keys if j in k]
#    candidates_p = [k for k in prosetta_keys if j in k]
#    if candidates_j and candidates_p:
#        return '-'.join(candidates_j[0]) + ' ||| ' + '-'.join(candidates_p[0])
#    else: return ""


'''
Applying document model on test corpus
'''

java_files_path = "..//test_data//Java"
python_files_path = "..//test_data//python"
alignment_file = '..//models//align-sample.txt'
java_pattern =  r'.*\.java'
python_pattern =  r'.*\.py'

def read_corpus(path,pattern):
    cr = PlaintextCorpusReader(path,pattern)
    for fid in cr.fileids():
        code = cr.raw(fid)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            tokens = [token for t in tokens if t.type == tokenize.NAME or t.type == tokenize.OP for token in t.string.split(" ")]
            if tokens:
                # print("Task: %s; Color: %s" % (programming_task.name, color_val))
                yield TaggedDocument(tokens, fid)
        except tokenize.TokenError as e:
            # print("%s: %s" % (type(e).__name__, e))
            pass
        except IndentationError as e:
            # print("%s: %s" % (type(e).__name__, e))
            pass
        except Exception as e:
            print("%s: %s" % (type(e).__name__, e))
            pass            
            
python_test_corpus = list(read_corpus(python_files_path,python_pattern))
print("Python Test corpus size: %s" % len(python_test_corpus))

java_test_corpus = list(read_corpus(java_files_path,java_pattern))
print("Java Test corpus size: %s" % len(java_test_corpus))



'''
Constructing model files for the different languages
'''

def write_model_to_file(model_pair,fname):
    f = open(fname,'w')
    lst = []
    for x,v in model_pair:
        sublst = []
        sublst.append(x.strip())
        sublst += [ str(s).strip() for s in v]
        lst.append(' '.join(sublst))
    f.write('\n'.join(lst))
    f.close()
    
jdocs = [(doc.tags, java_model.infer_vector(doc.words, steps=50)) 
                                        for doc in java_test_corpus]
pdocs = [(doc.tags, python_model.infer_vector(doc.words, steps=50)) 
                                        for doc in python_test_corpus]

java_pairs = [(x,java_model.docvecs[x]) for x in java_model.docvecs.doctags.keys()]
python_pairs = [(x,python_model.docvecs[x]) for x in python_model.docvecs.doctags.keys()]

full_jpairs = java_pairs + jdocs
full_pypairs = python_pairs + pdocs
        
write_model_to_file(full_jpairs,out_java_model)
write_model_to_file(full_pypairs,out_python_model)

'''
Constructing sample alignment seeds.

steps:
    load fnames from one language
    choose a random sample
    move to corresponding folder
    choose one file, mark as selected for removing redunduncy
    print the pair fn1 ||| fn2
'''


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
#    p = form_pair(s, selected)
    pairs.append(s + ' ||| ' + selected)
#    if p:
#        pairs.append(form_pair(s, selected))
    
with open(alignment_file,'w') as f:
    f.write("\n".join(pairs))


