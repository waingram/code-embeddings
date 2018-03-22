# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:26:48 2018

@author: Arnaoty
"""

from nltk.corpus import PlaintextCorpusReader
from javalanglib.javalang import tokenizer
import re


class Raw_Code_Reader(object):
    ''' Raw Code Reader is a class that reads in a repository of java code, preprocess the code, 
    tokenize it and split into different methods.
    TODO: this could be refactored to be a super class for several subclasses like JavaCodeReader, PythonCodeReader...etc
    '''
        
    def __init__(self, corpusDirectory):
        """The constructor initializes the corpus
        from a local directory path"""
        self.corpus_root = corpusDirectory
        self.corpusFiles = PlaintextCorpusReader(self.corpus_root, r'.*\.java')
        self.preprocessed_files = PlaintextCorpusReader(self.corpus_root, r'.*\.mthds')
        self.iterator = 0
    
        
    def load_all_data(self):
        '''load all java files under the corpus directory'''
        self.plaincorpus=[]
        for f in self.corpusFiles.fileids():
            try:
                self.plaincorpus += [self.corpusFiles.raw(f)]
            except:
                print(f + " can't be decoded")
        return self.plaincorpus


    def load_all_methods(self, method_separator = r"-{5,}MethodSeparator-{5,}"):
        ''' Assumes each java file is parsed for methods. Methods are extracted and written to new files with .mthds extension.
        Methods in the same mthds file are separated by some user defined separator (regular expression).
        The default separator is r"-{5,}MethodSeparator-{5,}" which matches text like ----------MethodSeparator-----------
        '''
        self.methods_corpus=[]
        for f in self.preprocessed_files.fileids():
            try:
                self.methods_corpus += [re.split(method_separator, self.preprocessed_files.raw(f))]
            except:
                print(f + " can't be decoded")
        return self.methods_corpus


    def load_next_batch(self, batch_size=1000):
        '''load next batch of java files under the corpus directory given by the batch size param'''
        if self.iterator >= len(self.corpusFiles.fileids()):
            print("No more data to load....")
        
        self.plaincorpus=[]
        for f in self.corpusFiles.fileids()[self.iterator: self.iterator+ batch_size]:
            try:
                self.plaincorpus += [self.corpusFiles.raw(f)]
            except:
                print(f + " can't be decoded... continue decoding")
        self.iterator = self.iterator+ batch_size
        return self.plaincorpus
  

    def load_next_batch_methods(self, batch_size=1000, method_separator = r"-{5,}MethodSeparator-{5,}"):
        '''load next batch of mthds files under the corpus directory given by the batch size param'''
        if self.iterator >= len(self.preprocessed_files.fileids()):
            print("No more data to load....")
        
        self.methods_corpus=[]
        for f in self.preprocessed_files.fileids()[self.iterator: self.iterator+ batch_size]:
            try:
                self.methods_corpus += [re.split(method_separator, self.preprocessed_files.raw(f))]
            except:
                print(f + " can't be decoded... continue decoding")
        self.iterator = self.iterator+ batch_size
        return self.methods_corpus
        
        
    def tokenize_program(self, code):
        'Tokenize any snippet of java code'
        tokens = []
        try:
            ts = list(tokenizer.tokenize(code))
            tokens = [tkn for t in ts for tkn in t.value.split(" ")]
        except:
            print("Error tokenizing the program -> " + code[:100] )
        return tokens
        
    'Works poorly with anynomous class methods or methods without scope modifiers'
    'That is why I no longer use this. I split methods using changeDistiller java code.' 
    'I then write the methods to file system and read it here using load_methods functions'
    def extract_methods(self, code):
#       x =re.findall(r'(?:public|private|protected|static)(?:\s|.)+?{(?:\s|.)+?}',rw)
        methods = []
        arr = re.split("(?:public|private|protected|static)",code)
        methods = [m[:m.rfind('}')+1].strip() for m in arr                     
                    if m.strip() and not m.strip().startswith(('class','interface','import', 'package')) ]
        methods = [m for m in methods if m.strip()]
        return methods
        
    
    def tokenize_loaded_data(self):
        'Tokenize all the loaded java code'
        self.tokens = [self.tokenize_program(p) for p in self.plaincorpus]
        return self.tokens
        
    def tokenize_loaded_methods(self):
        'Tokenize all the loaded methods'
        self.tokens = [[self.tokenize_program(m) for m in p]
                            for p in self.methods_corpus]
        return self.tokens
    
    def has_next(self):
        return self.iterator < len(self.corpusFiles.fileids())
    
