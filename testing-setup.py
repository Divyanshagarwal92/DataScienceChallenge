import scipy
import numpy
import math
import classifier_set as cs

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



if __name__ == '__main__':
	X = numpy.load('features.npy')
	Y = numpy.load('labels.npy')
	cs.RF(X,Y,15)



