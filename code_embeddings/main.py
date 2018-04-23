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
import random
from embedding_visualizer import tsne_learn, plot_model, plot_model_3d
#from gensim.models import Doc2Vec


'''
This part of the code just generates embeddingds on the word level
'''
# batch_size = 42120 #10000
#corpusDirectory = "E:\\PHD\\Thesis VT\\Data\\era_bcb_sample"
corpusDirectory = "E:\\PHD\\Thesis VT\\SourceCodeEmbedding\\code-embeddings\\test_data\\Java"
code_repository = Raw_Code_Reader(corpusDirectory)
programs_batch = code_repository.load_all_data()
# programs_batch = code_repository.load_next_batch(batch_size)
tokenized_code = code_repository.tokenize_loaded_data()

print('Data Loaded')
print('Start training')
wrd_embd = Word_Embedder(tokenized_code,'w2v', size=100, window=10, min_count=1, iter = 5, workers=4)
wrd_embd.save_model('word2v.mdl')



#out = open('rosettaJavaCorpus.src','w')
#for m in tokenized_code:
#    for x in m:
#        try:
#            out.write(x)
#            out.write(' ')
#        except:
#            continue
#            
#    out.write('\n')
#out.close()

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
#
#my_list = ['--','count','delete','HttpGet', 'NullPointerException','BufferedInputStream','FileOutputStream','FileInputStream',
#           'FileReader','File','file','BufferedOutputStream','ObjectInputStream','Object','obj', 'RandomAccessFile', 'extends',
#           'implements','baseClass','abstract','class','x','count','getName','y','X','Y','z','xx','yy','i','j','k','for','while',
#           'TODO','if','px','Point','point','height','getWidth','getHeight','getImageWidth','imageHeight','0x0f','0xf','404',
#           'HTTP_NOT_FOUND','SC_NOT_FOUND','403','5','0','.','length','size','7','<<','>>>=','readBits','skipBytes','getBits',
#           '>>=','true','false','flag','boolean','visible','showOpenAndClose','getButtonReplay','isWritable','isValid','isComplete',
#           'isReadOnly','isReadable','hasChanged','SortedSet','List','HashSet','TreeSet','Set','HashMap','HashTable','LinkedList',
#           'Map','LinkedList','ArrayList','Collection','Arrays','Vector','Iterable','Comparator','Comparable','pause','wait','run',
#           'Runnable','HttpClient','HttpGet','=','==','!=','>=','>','<','GetMethod','HttpPost','HttpResponse','HttpHead','HttpPut',
#           'HttpDelete','equalsIgnoreCase','instanceof','equals','startsWith','endsWith','contains','containsIgnoreCase','null',
#           'contentEquals','compareToIgnoreCase','IndexOutOfBoundsException','IllegalArgumentException','ArrayIndexOutOfBoundsException',
#           'ClassCastException','AssertionError','IOException','NoSuchElementException','SecurityException','RuntimeException','*',
#           ']','}',')','(','{','(','args','params','++','+','+=','--','-=','&','^','/','java','import','interface', 'public', 'private',
#           'protected','Parser','SAXParser','DomParser','XMLReader','package','button','<p>','jar','DOM','localhost'
#           ]



#v13,l13 = iso_map_learn(mdl2,my_list,1,num_components=2)
#
#plot_model(v13,l13,1)
#v13,l13 = iso_map_learn(mdl2,my_list,1,num_components=3)
#plot_model(v13,l13,1)
#plot_model_3d(v13,l13,1)
#v13,l13 = tsne_learn(mdl2,my_list,1,num_components=3)
#plot_model_3d(v13,l13,1)
#plot_model(v13,l13,1)

mdl2 = wrd_embd.get_model()
mdl2.wv.most_similar('--')
mdl2.wv.most_similar('count')
mdl2.wv.most_similar('delete')
mdl2.wv.most_similar('HttpGet')
mdl2.wv.most_similar('NullPointerException')
#tsne_plot(mdl)



index_file = "E://PHD//Thesis VT//SourceCodeEmbedding//code-embeddings//models//autoEncode models//vocab.txt"
w_embed_file = "E://PHD//Thesis VT//SourceCodeEmbedding//code-embeddings//models//autoEncode models//embed.txt"



f = open(w_embed_file,'r')
embed = f.readlines()
embd = [e.strip().split() for e in embed]
embed = [[float(n) for n in e] for e in embd]
f.close()

f = open(index_file,'r')
indx = f.readlines()
indx = [i.strip() for i in indx]
f.close()

'building a dictionary of word2vectors and passing it for visualization'
word_model = dict(zip(indx,embed))
def get_word_embedding(word):
    return word_model[word]
#    return (embed[indx.index(word)])

#lst = [a for a in my_list if a in word_model.keys()]
#v,l = tsne_learn(word_model,lst,1)
#plot_model(v,l,1)

keys = random.sample(list(word_model), 120)
v,l = tsne_learn(word_model,keys,1)
plot_model(v,l,1)

v,l = tsne_learn(word_model,word_model.keys(),1)
plot_model(v,l,0.02)

v,l = tsne_learn(word_model,word_model.keys(),1,3)
plot_model_3d(v,l,0.02)

#tsne_plot(word_model)

#import os
#for filename in os.listdir(samples_file):
#    f = '00' + filename
#    f = f[-9:]
#    print(filename)
#    if f != filename:
#        os.rename(os.path.join(samples_file,filename), os.path.join(samples_file,f))




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