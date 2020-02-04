# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:17:12 2020

@author: janz
"""

import warnings
import loadData
from trainClassifiers import trainClassifiers
from decisionFusion import decisionFusion
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from optimizeEnsemble import optimizeEnsemble
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
warnings.filterwarnings('ignore')

def mainProgram(dataset):
    # load data
    X, Y = loadData.load_dataset(dataset)
    fold = 10
    kf = KFold(n_splits=fold)
    current_fold = 0
    acc = 0
    optimized_acc = 0
    
    # clusters
    X_train_clusters = []
    Y_train_clusters = []
    for train, test in kf.split(X):
    	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
    	X_train = preprocessing.normalize(X_train)
    	X_test  = preprocessing.normalize(X_test)        
    	sil = []
        #dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    	K = range(2,20);
    	for k in K:
            kmeans = KMeans(n_clusters = k).fit(X_train)
            labels = kmeans.labels_
            sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))
    	optimum_K = K[sil.index(max(sil))]
    	plt.plot(K, sil, 'bx-')
    	plt.xlabel('k')
    	plt.ylabel('Silhouette Score')
    	plt.title('Silhouette Dissimilarity Scores for various k')
    	plt.axvline(x=optimum_K, color='r', linestyle='--')
    	fig1 = plt.gcf()
    	plt.show()
    	plt.draw()
    	fig1.savefig(dataset+'.png', format='png', bbox_inches='tight',dpi=300)
    	plt.close()
    	kmeans = KMeans(n_clusters = optimum_K, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)			
    	kmeans.fit(X_train)
    	count = len(np.unique(Y_train))
    	for j in range(optimum_K):
            X_train_temp = X_train[kmeans.labels_==j]
            Y_train_temp = Y_train[kmeans.labels_==j]
            v = len(np.unique(Y_train_temp))
            if v > 1:
                X_train_clusters.append(X_train_temp)
                Y_train_clusters.append(Y_train_temp)  
    	count = int(len(X_test)/5)
    	valX = X_test[0:count]
    	valy = Y_test[0:count]
    	X_test = X_test[count:]
    	Y_test = Y_test[count:]  	
    	
    	ensemble = trainClassifiers(X_train_clusters, Y_train_clusters)
    	acc += decisionFusion(ensemble, X_test, Y_test)        
    	
    	optimized_ensemble = optimizeEnsemble(ensemble, valX, valy)
    	optimized_acc += decisionFusion(optimized_ensemble, X_test, Y_test)        
    	current_fold += 1
    	print ("Non_optimized and Optimized Accuracy for " + dataset  + " is: " + str(acc/current_fold) + " and " + str(optimized_acc/current_fold))   
    return ((acc/current_fold), (acc/current_fold))
      
