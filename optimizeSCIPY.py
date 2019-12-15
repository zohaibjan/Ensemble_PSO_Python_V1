# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:23:09 2019

@author: janz
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:18:58 2019

@author: janz
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pyswarms as ps

def optimizeEnsemble(ensemble, valX, valy):
    ### START OF OBJECTIVE FUNCTION ###
    def objective(indexOfClasifiers):
        error = np.zeros(len(valX))
        predictions = pd.DataFrame()
        index = 0
        record_change = []
        changes = indexOfClasifiers%1
        for i in range(len(changes)):
            if changes[i] != 0:
                record_change.append(i)
        for eachChange in record_change:
            indexOfClasifiers[eachChange] = 1
        for i in range(len(indexOfClasifiers)):   
            if indexOfClasifiers[i] > 0.5:
                predictions.insert(index,column = index, value = ensemble[i].predict(valX))
                index += 1
            else:
                continue
        results = predictions.mode(axis='columns')[0]
        results = np.asarray(results)
        error = 0
        for i in range(len(results)):
            if (int(results[i]) != valy[i]):
                error += 1
        error = error/len(results)
        print("Current error is :" + str(error))
       # print("Minimizing current error : " + str(error))
        return error
    ### END OF OBJECTIVE FUNCTION ###
    
    ### SOME PARAMETERS FOR OPTIMIZATION
    
    x0 = np.random.choice([0,1],size=(1,len(ensemble)))
    b = (0, 1)
    bnds = (b,)*len(ensemble)
	#cons = ({'type': 'eq', 'fun': lambda x:  x.sum() - 1.0})
    sol = minimize(objective, x0, method = 'Nelder-Mead' , bounds = bnds,\
                   options={"maxiter": 5000, 'disp' : True, 'xatol': 0.1, 'maxfev': 1000})
 
    for i in range(len(sol.x)):
        if np.round(sol.x[i]) == 1:
            optimized.append(ensemble[i])
    
    
    return optimized
  

