import warnings
import loadData
from trainClassifiers import trainClassifiers
from decisionFusion import decisionFusion
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from optimizeEnsemble import optimizeEnsemble
warnings.filterwarnings('ignore')

def mainProgram(dataset):
    # load data
    X, Y = loadData.load_dataset(dataset)
    fold = 10
    num_of_clusters = 3
    kf = KFold(n_splits=fold)
    current_fold = 0
    acc = 0
    optimized_acc = 0
    
    # clusters
    X_train_clusters = []
    Y_train_clusters = []
    for train, test in kf.split(X):
    	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
    	# for k clusters
    	for i in range(num_of_clusters):
    		if i == 0:
    			continue
    		kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)			
    		kmeans.fit(X_train)
    		for j in range(i):
    			X_train_temp = X_train[kmeans.labels_==j]
    			Y_train_temp = Y_train[kmeans.labels_==j]
    			v, c = np.unique(Y_train_temp, return_counts=True)	
    			if v.shape[0] < 2 or 1 in c:
    				print('Cluster ' + str(j) + ' only has one class: ' + str(v[0]))
    			else:
    				X_train_clusters.append(X_train_temp)
    				Y_train_clusters.append(Y_train_temp)                 
    	breakpoint            
    	count = int(len(X_test)/10)
    	valX = X_test[0:count]
    	valy = Y_test[0:count]
    	X_test = X_test[count:]
    	Y_test = Y_test[count:]  	
    	print("Now training classifiers")
    	ensemble = trainClassifiers(X_train_clusters, Y_train_clusters)
    	acc += decisionFusion(ensemble, X_test, Y_test)        
    	print("Now running Particle Swarm Optimization")
    	optimized_ensemble = optimizeEnsemble(ensemble, valX, valy)
    	optimized_acc += decisionFusion(optimized_ensemble, X_test, Y_test)        
    	current_fold += 1
        
    return ((acc/current_fold), (optimized_acc/current_fold))
      