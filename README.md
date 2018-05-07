# A Comparative Study of Various Code Embeddings in Software Semantic Matching

## Abstract

The ability to search code repositories for functionally equivalent code would be a tremendous benefit to software engineering. Code reuse is fundamental to software engineering, and open source code repositories have become rich sources of reusable code. In this study, we examine how machine learning techniques used in Natural Language Processing (NLP) for representing words and documents as vectors can be applied to representing code fragments in vector space. To do so, we amass a large corpus of programming tasks implemented in multiple programming languages. We then apply existing document embedding techniques to our corpus of code so that we can map each code fragment to a point in vector space and study to what extent these document embeddings are useful in capturing the semantics of software code. Finally we design and implement a code-matching application for locating functionally equivalent code fragments based on vector embeddings and use this application for evaluating the different embeddings.

## Requirements

 - astor
 - Flask
 - gensim
 - javalang
 - matplotlib
 - regex
 - sklearn

Install packages with ```pipenv```:

    $ pipenv install

## Experiments

 - [Doc2Vec Java embedding](https://github.com/waingram/code-embeddings/blob/master/experiments/doc2vec_experiments.ipynb)
 - [Doc2Vec Python embedding](https://github.com/waingram/code-embeddings/blob/master/experiments/doc2vec_python_experiments.ipynb)
 - [Java Parsing](https://github.com/waingram/code-embeddings/blob/master/experiments/java_parsing.ipynb)
 - [Python Parsing](https://github.com/waingram/code-embeddings/blob/master/experiments/python_parsing.ipynb)