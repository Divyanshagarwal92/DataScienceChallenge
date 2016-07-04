import scipy
import numpy
import math

from os import listdir
from os.path import isfile, join
from os import walk

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt

def myPreprocessing( X_train, X_test):
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train_transformed = scaler.transform(X_train)
	X_test_transformed = scaler.transform(X_test)
	return [X_train_transformed, X_test_transformed]

def SVM( X, Y):
	print 'SVM'
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
			X, Y, test_size=0.4, random_state=0)
	clf = svm.SVC().fit(X_train, Y_train)
	print "Score: " + "{0:.3f}".format(clf.score(X_test, Y_test))


def scaledSVM_RBF( X, Y, gamma):
	[X_train_transformed, X_test_transformed] = myPreprocessing( X, X)
	rbf_svc = svm.SVC( kernel='rbf', gamma=gamma,  C=1)
	scores = cross_validation.cross_val_score( rbf_svc, X_train_transformed, Y, cv=5)
	print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()
	
def adaptiveBoosting(X,Y, n_estimators):
	[X_train_transformed, X_test_transformed] = myPreprocessing( X, X)
	ab = AdaBoostClassifier(n_estimators=n_estimators)
	scores = cross_validation.cross_val_score( ab, X_train_transformed, Y, cv=5)
	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()

def DecisionTree(X,Y):
	[X_train_transformed, X_test_transformed] = myPreprocessing( X, X)
	dt = tree.DecisionTreeClassifier()
	scores = cross_validation.cross_val_score( dt, X_train_transformed, Y, cv=5)
	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()

def RF( X, Y, num):
	[X_train_transformed, X_test_transformed] = myPreprocessing( X, X)
	rf = RandomForestClassifier(n_estimators=num)
	scores = cross_validation.cross_val_score( rf, X_train_transformed, Y, cv=5, scoring = 'f1')
	print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
	return scores.mean()

if __name__ == '__main__':
	
	classifier_list = ['svm_rbf', 'adaboost', 'random_forest', 'decision_tree']
	classifier_list = ' '.join(classifier_list)
	classifier = raw_input('Pick a classifier: ' + classifier_list + '\n')	

	X = numpy.load('features.npy')
	Y = numpy.load('labels.npy')
	dat = [ X, Y]
	parameters = []
	score = []
	
	if classifier == 'svm_rbf':
		print 'Scaled SVM with RBF kernel'
		gamma = 0.001
		for i in range(5):
			parameters.append(math.log10(gamma))
			print 'Gamma: ' + str(gamma)
			score.append(scaledSVM_Sigmoid( dat[0], dat[1], gamma))
			gamma = 10*gamma
		plt.plot( parameters, score)
		plt.xlabel('Gamma (Log base 10)')
		plt.ylabel('Score')
		plt.title('Accuracy of SVM - RBF kernel')
		plt.grid(True)
		plt.show()
		
	if classifier == 'adaboost':
		print 'Adaptive Boosting'
		for i in range(1,401,40):
			parameters.append(i)
			score.append(adaptiveBoosting(dat[0],dat[1], i))
		plt.plot( parameters, score)
		plt.xlabel('Number of Estimators')
		plt.ylabel('Score')
		plt.title('Accuracy of Adaptive Boosting')
		plt.grid(True)
		plt.show()
	
	if classifier == 'random_forest':
		print 'Random Forest'
		params = [1,5,10,15,20]
		for i in params:
			print 'Num of Estimators: ' + str(i)
			score.append( RF(dat[0],dat[1],i))
		plt.plot( params, score)
		plt.xlabel('Num. of Estimators')
		plt.ylabel('Score')
		plt.title('Accuracy of Random Forest')
		plt.grid(True)
		plt.show()
