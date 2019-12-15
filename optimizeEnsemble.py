# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:18:58 2019

@author: janz
"""
import numpy as np
import pandas as pd
import pyswarms as ps

def optimizeEnsemble(ensemble, valX, valy):
    ### START OF OBJECTIVE FUNCTION ###
    def f_per_particle(m, a):
        total_features = len(ensemble)
        index = 0
        predictions = pd.DataFrame()
        # Perform classification and store performance in P
        for i in range(len(m)):   
            if m[i] == 1:
                predictions.insert(index,column = index, value = ensemble[i].predict(valX))
                index += 1
            else:
                continue
        results = predictions.mode(axis='columns')[0]
        results = np.asarray(results)
        acc = 0
        for i in range(len(results)):
            if (int(results[i]) == valy[i]):
                acc+= 1
        P = acc/len(valy)
        #j = (a * (1.0 - P) + (1.0 - a) * (1 - (index / total_features)))
        j=1-P
       # print("Minimizing current error : " + str(error))
        return j
    
    def f(x, alpha=0.88):
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)   
    ### END OF OBJECTIVE FUNCTION ###
    
    ### SOME PARAMETERS FOR OPTIMIZATION
    optimized = []       
    # Initialize swarm, arbitrary
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
 
    # Call instance of PSO
    dimensions = len(ensemble) # dimensions should be the number of Classifiers being optimized
    optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=10)
    print("Optimization process completed with a cost of : " + str(cost))
    
    for i in range(len(pos)):
        if pos[i] == 1:
            optimized.append(ensemble[i])
    return optimized
  

