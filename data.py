import bson
import nltk
import re
from nltk.corpus import stopwords
from gensim import models
import numpy as np

num_features = 300

def review_to_words( raw_response, timeTokenList ):
    
    raw_response = raw_response.replace('.','')
    
    #get unique time tokens
    entities = set()
    for t in timeTokenList:
        entities.add(t)
        
    #reassign unique time tokens list to timeTokenList
    timeTokenList = entities
    
    entities = ' '.join(entities)
    time_words = entities.split()
    
    words = raw_response.split()
    #remove words in timeTokenList
    filter1_words = [w for w in words if not w in time_words]
    raw_response = ' '.join(filter1_words)
    
    #remove numbers
    letters_only = re.sub("[^a-zA-Z]", " ", raw_response)        
    words = letters_only.split()     

    #remove scrambled words like PERSON, TIME, LOCATION, etc
    filter_words = [ w for w in words if not w.isupper() ]
    
    #remove stop-words
    stops = set(stopwords.words("english"))                    
    meaningful_words = [w for w in filter_words if not w in stops]
    
    return [" ".join( meaningful_words ).lower(), timeTokenList]



def loadData( filename):
	b_file = open( filename, 'rb')
	bs = b_file.read()
	dictionary = bson.decode_all(bs)
	return dictionary

def getFeatures(dic, w):
	#dic = list_dictionary[i]
	sentence = dic['dataPoint']['smearedSentence']
	timeToken = dic['dataPoint']['timeEntityTokens']
	[modified_sentence, timeToken] = review_to_words(sentence, timeToken)
	words = modified_sentence.split()
	sumVec = []
	cnt = 0
	flag = 0
	num_words = len(words)
	#print 'Length: ' + str( len(words))
	for i in range( len(words)):
		word = words[i]
		#print word
		if word in w and word != '':
			vec = w[word]
			if vec.size == 0:
				#print 'empty array'
				continue
			cnt=cnt+1
			if flag == 0:
				sumVec = vec
				flag = 1
			else:
				sumVec = np.add( sumVec, vec)
	#print 'num words: ' + str(cnt)
	# if no word present, assume a positive time entity, done implicityly by assigning the works keywoard
	if cnt == 0:
		avg_vector = w['works']
	else:
		avg_vector = np.divide(sumVec, cnt)
	return avg_vector

def getAllAvgFeatures( list_dictionary, model):
	counter = 0
	emailFeatureVecs = np.zeros((len(list_dictionary), num_features),dtype="float32")
	emailLabel = np.zeros(len(list_dictionary),dtype="int32" )
	countP = 0
	countN = 0
	for dic in list_dictionary:
		label = dic['dataPoint']['label']
		emailFeatureVecs[counter] = getFeatures(dic, model)
		
		if label == 'POSITIVE_TIME':
			emailLabel[counter] = 1
			countP = countP + 1
		if label == 'NEGATIVE_TIME':
			#print dic['dataPoint']['smearedSentence']
			#print counter
			#print label
			#emailLabel[counter] = 0
			#print emailLabel[counter]
			countN = countN + 1
		counter = counter + 1
	print 'Positive time entities ' + str(countP)
	print 'Negative time entities ' + str(countN)
	return [ emailFeatureVecs, emailLabel]

if __name__ == '__main__':
	w = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	print 'loading word2vec'
	filename = "PositiveNegativeTimeExerciseDataSet.bson"
	list_dictionary = loadData(filename)
	[X,Y] = getAllAvgFeatures( list_dictionary, w)
	np.save('features',X)
	np.save('labels',Y)

