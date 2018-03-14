# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:26:48 2018

@author: Arnaoty
"""

from nltk.corpus import PlaintextCorpusReader
from javalanglib.javalang import tokenizer

class Raw_Code_Reader(object):
    
    def __init__(self, corpusDirectory):
        """The constructor initializes the sentiment corpus
        from a local directory path"""
        self.corpus_root = corpusDirectory
        self.corpusFiles = PlaintextCorpusReader(self.corpus_root, r'.*\.java')
        self.iterator = 0
        
        
        
    def load_all_data(self):
        self.plaincorpus=[]
        for f in self.corpusFiles.fileids():
            try:
                self.plaincorpus += [self.corpusFiles.raw(f)]
            except:
                print(f + " can't be decoded")

    def load_next_batch(self, batch_size=1000):
        if self.iterator >= len(self.corpusFiles.fileids()):
            print("No more data to load....")
        
        self.plaincorpus=[]
        for f in self.corpusFiles.fileids()[self.iterator: self.iterator+ batch_size]:
            try:
                self.plaincorpus += [self.corpusFiles.raw(f)]
            except:
                print(f + " can't be decoded... continue decoding")
        self.iterator = self.iterator+ batch_size
        
    def tokenize_program(self, code):
        tokens = []
        try:
            ts = list(tokenizer.tokenize(code))
            tokens = [tkn for t in ts for tkn in t.value.split(" ")]
        except:
            print("Error tokenizing the program -> " + code[:100] )
        return tokens
        

    def tokenize_loaded_data(self):
        self.tokens = [self.tokenize_program(p) for p in self.plaincorpus]
        return self.tokens
        
    def has_next(self):
        return self.iterator < len(self.corpusFiles.fileids())
#    def split_to_functions(self):
#        tokens = list(tokenizer.tokenize('System.out.println("Hello hello" + "world");\n c=34.6+5; '))