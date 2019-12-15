# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:44:49 2019

@author: janz
"""
import numpy as np
import pandas as pd

def decisionFusion (ensemble, X, y):
    acc = np.zeros(len(X))
    predictions = pd.DataFrame()
    for i in range(len(ensemble)):    
        predictions.insert(i,column = i, value = ensemble[i].predict(X))
    results = predictions.mode(axis = 'columns')[0]
    results = np.asarray(results)        
    for i in range(len(results)):
        if (int(results[i]) == y[i]):
            acc[i] = 1
        else:
            acc[i] = 0
    return np.mean(acc)