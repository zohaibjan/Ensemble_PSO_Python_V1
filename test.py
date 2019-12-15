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
file = open("results.csv", mode="a")
results = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
results.writerow(["Data set", "Accuracy without optimization","Accuracy with optimization"])
file.close();
