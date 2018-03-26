# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:58:55 2018

@author: Arnaoty
"""
from sklearn.manifold import TSNE, Isomap
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from random import uniform

def tsne_learn(model,  train_percent = 0.01, num_components=2):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        u = uniform(0,1)
        if u < train_percent :
            tokens.append(model[word])
            labels.append(word)
    
    print("selecting random percentage of the data: " + str(len(tokens)) + "examples selected.")
    tsne_model = TSNE(perplexity=40, n_components=num_components, init='pca', n_iter=2500, random_state=23)
    
    print("Training TSNE model ")
    new_values = tsne_model.fit_transform(tokens)
    return new_values, labels

def iso_map_learn(model,  train_percent = 0.01, num_components=2):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        u = uniform(0,1)
        if u < train_percent :
            tokens.append(model[word])
            labels.append(word)
    
    print("selecting random percentage of the data: " + str(len(tokens)) + "examples selected.")
    iso_model = Isomap(n_components=num_components)
    
    print("Training ISO_Map model ")
    new_values = iso_model.fit_transform(tokens)
    return new_values, labels

    
def plot_model(new_values, labels, show_percent = 0.1):
    x = []
    y = []
    for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        u = uniform(0,1)
        if u < show_percent :
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.show()
    
def plot_model_3d(new_values, labels, show_percent = 0.1):
    x = []
    y = []
    z = []
    for value in new_values:
            x.append(value[0])
            y.append(value[1])
            z.append(value[1])
    
    fig = plt.figure(figsize=(16, 16))
    ax = Axes3D(fig)
        
#    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        u = uniform(0,1)
        if u < show_percent :
            ax.scatter(x[i],y[i],z[i])
            ax.text(x[i],y[i],z[i],'%s' % (labels[i]), size=10, zorder=1, color='k')
#            plt.annotate(labels[i],
#                         xyz=(x[i], y[i]),
#                         xytext=(5, 2),
#                         textcoords='offset points',
#                         ha='right',
#                         va='bottom')
    plt.show()
    