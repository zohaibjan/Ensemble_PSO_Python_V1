# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:12:00 2019

@author: janz
"""

from mainProgram import mainProgram
import numpy as np
import csv
import os.path

data = {'thyroid','wine','diabetes',\
        'segment', 'ecoli','cancer',\
        'vehicle','iris','liver','ionosphere',\
        'sonar','glass'}

numOfRuns = 1
for dataset in data:
    if os.path.isfile('results.csv'):
        file = open("results.csv", mode="a")
        results = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    else:
        file = open("results.csv", mode="w")
        results = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        results.writerow(['dataset', 'accuracy','optimized_accuracy'])
    accuracy = np.zeros(numOfRuns)
    optimized_accuracy = np.zeros(numOfRuns)
    results.writerow(['dataset', 'accuracy','optimized_accuracy'])
    for i in range(numOfRuns):
        temp = mainProgram(dataset)
        accuracy[i] = temp[0]
        optimized_accuracy = temp[1]
    results.writerow([dataset, str(np.mean(accuracy)),str(np.mean(optimized_accuracy))])
file.close();
print ("Non_optimized and Optimized Accuracy for " + dataset  + " is: " + str(np.mean(accuracy)) + " and " + str(np.mean(optimized_accuracy)))
