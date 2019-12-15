# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:44:49 2019

@author: janz
"""
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def trainClassifiers (X, y):
    classifiers = [];    
    for i in range(len(X)):    
        #Train SVM Clasifier
        model = svm.SVC(gamma = 0.001, decision_function_shape='ovo')
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
        
        #Train Decision Tree Classifier
        model = tree.DecisionTreeClassifier()
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))

        #Naive Bayes Classifier
        model = GaussianNB()
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
        
        #Multi layer perceptron Classifier
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
        
        #KNN Classifier
        model  = KNeighborsClassifier(n_neighbors=3)
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
        
        #Linear Discriminant Analysis Classifier
        model = LinearDiscriminantAnalysis()
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
		
		#Logistic Regression Classifier
        model = LogisticRegression()
        classifiers.append(model.fit(X[i].astype(float), y[i].astype(float)))
		
    return classifiers