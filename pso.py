import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


class PSO(object):
	def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value, X_train, X_test, Y_train, Y_test, classifiers):
		self.particle_num = particle_num
		self.particle_dim = particle_dim
		self.iter_num = iter_num
		self.c1 = c1
		self.c2 = c2
		self.w = w    
		self.max_value = max_value
		self.min_value = min_value
		self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
		self.classifiers = classifiers
        
	def swarm_origin(self):
		particle_loc = []
		particle_dir = []
		for i in range(self.particle_num):
			tmp1 = []
			tmp2 = []
			for j in range(self.particle_dim):
				a = random.random()
				b = random.random()
				tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
				tmp2.append(b)
			particle_loc.append(tmp1)
			particle_dir.append(tmp2)
		return particle_loc,particle_dir
 
	def fitness(self, particle_loc):
		fitness_value = []
		for i in range(self.particle_num):
			voting_classifier = VotingClassifier(estimators=self.classifiers, voting = 'soft', weights = particle_loc[i])
			cv_scores = cross_val_score(voting_classifier,self.X_train,self.Y_train,cv =3,scoring = 'accuracy')
			fitness_value.append(cv_scores.mean())
		current_fitness = 0.0
		current_parameter = []
		for i in range(self.particle_num):
			if current_fitness < fitness_value[i]:
				current_fitness = fitness_value[i]
				current_parameter = particle_loc[i]
		return fitness_value, current_fitness, current_parameter 

	def update(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
		for i in range(self.particle_num): 
			a1 = [x * self.w for x in particle_dir[i]]
			a2 = [y * self.c1 * random.random() for y in list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
			a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
			particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
			particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))

		parameter_list = []
		for i in range(self.particle_dim):
			tmp1 = []
			for j in range(self.particle_num):
				tmp1.append(particle_loc[j][i])
			parameter_list.append(tmp1)
		value = []
		for i in range(self.particle_dim):
			tmp2 = []
			tmp2.append(max(parameter_list[i]))
			tmp2.append(min(parameter_list[i]))
			value.append(tmp2)
        
		for i in range(self.particle_num):
			for j in range(self.particle_dim):
				particle_loc[i][j] = (particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (self.max_value - self.min_value) + self.min_value
                
		return particle_loc, particle_dir

	def round_paras(self, v):
		for i in range(len(v)):
			if v[i] >= 0.5:
				v[i] = 1
			else:
				v[i] = 0
		return v
       
	def run(self):
		results = []
		best_fitness = 0.0 
		particle_loc,particle_dir = self.swarm_origin()
		gbest_parameter = []
		for i in range(self.particle_dim):
			gbest_parameter.append(0.0)
		pbest_parameters = []
		for i in range(self.particle_num):
			tmp1 = []
			for j in range(self.particle_dim):
				tmp1.append(0.0)
			pbest_parameters.append(tmp1)
		fitness_value = []
		for i in range(self.particle_num):
			fitness_value.append(0.0)
		for i in range(self.iter_num):
			current_fitness_value,current_best_fitness,current_best_parameter = self.fitness(particle_loc)
			for j in range(self.particle_num):
				if current_fitness_value[j] > fitness_value[j]:
					pbest_parameters[j] = particle_loc[j]
			if current_best_fitness > best_fitness:
				best_fitness = current_best_fitness
				gbest_parameter = current_best_parameter
			results.append(best_fitness)
			fitness_value = current_fitness_value
			particle_loc, particle_dir = self.update(particle_loc, particle_dir, gbest_parameter, pbest_parameters)
		results.sort()
		gbest_parameter = self.round_paras(gbest_parameter)
		return results[-1], gbest_parameter

def pso_optimize(X_train, X_test, Y_train, Y_test, number_of_particles, dimensions, options, bounds, classifiers):
	particle_num = number_of_particles
	particle_dim = dimensions
	iter_num = options['itr']
	c1 = options['c1']
	c2 = options['c2']
	w = options['w']
	max_value = bounds['max']
	min_value = bounds['min']
	pso = PSO(particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value, X_train, X_test, Y_train, Y_test, classifiers)
	return pso.run()

def pso_optimize_clusters(number_of_particles, dimensions, options, bounds, X_train_clusters, Y_train_clusters, classifiers, X_test, Y_test):
	acc_sum = 0
	count = 0
	for i in range(X_train_clusters.shape[0]):
		_, current_best_parameter = pso_optimize(X_train_clusters[i], X_test, Y_train_clusters[i], Y_test, 
					number_of_particles, dimensions, options, bounds, classifiers)
		optimized_classifiers = []
		for j in range(len(current_best_parameter)):
			if current_best_parameter[j] == 1:
				optimized_classifiers.append(classifiers[j])
		voting_classifier = VotingClassifier(estimators = optimized_classifiers, voting = 'soft')
		voting_classifier.fit(X_train_clusters[i], Y_train_clusters[i])
		acc = voting_classifier.score(X_test, Y_test)
		acc_sum += acc
		count += 1
	return acc_sum/count
	


