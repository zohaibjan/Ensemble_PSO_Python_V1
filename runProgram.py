# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:12:00 2019

@author: janz
"""

from mainProgram import mainProgram
import numpy as np
import csv


data = {'thyroid','wine','diabetes',\
        'segment', 'ecoli','cancer',\
        'vehicle','iris','liver','ionosphere',\
        'sonar','glass'}
numOfRuns = 10
file = open("results.csv", mode="w")
for dataset in data:
    accuracy = np.zeros(numOfRuns)
    optimized_accuracy = np.zeros(numOfRuns)
    for i in range(numOfRuns):
        temp = mainProgram(dataset)
        accuracy[i] = temp[0]
        optimized_accuracy = temp[1]
    results = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results.writerow([dataset, str(np.mean(accuracy)),str(np.mean(optimized_accuracy))])
file.close();
print ("Non_optimized and Optimized Accuracy for " + dataset  + " is: " + str(np.mean(accuracy)) + " and " + str(np.mean(optimized_accuracy)))
