import pickle
import sys

from time import time
from collections import defaultdict, Counter

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

import gensim
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phraser

from scipy import spatial
import numpy as np

from parsexml import getEmbedding

# Gets similarity between a query vector and a document vector
def getSimilarity(queryEmb, docVec, model):
	return [1 - (spatial.distance.cosine(queryEmb, docVec))]

# Returns the topN documents by token relevance (vector model)
def getBestResults(queryStr, topN, tfidfMatrix):
	query = queryStr.split(" ")
	res = defaultdict(float)

	# Tokenizing and stemming query
	tquery = set()
	for word in query:
		word = stemmer.stem(word.lower())
		if word not in stopWords and len(word) > 2:
			tquery.add(word)

	# Computing average tf-idf score for each document
	for word in tquery:
		if word in tfidfMatrix:
			for doc in tfidfMatrix[word]:
				res[doc] += tfidfMatrix[word][doc]

	return dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:topN])

# Returns the topN documents by token relevance (word2vec model)
def getBestResultsWordEmbeddings(queryStr, topN, model, docVec, bigramModel):
	query = queryStr.split(" ")
	res = defaultdict(float)

	# Tokenizing and stemming query
	_tquery = list() # list and not set to preserve token order
	for word in query:
		word = stemmer.stem(word.lower())
		if word not in stopWords and len(word) > 2:
			_tquery.append(word)
	
	# Finding query's bigrams
	tquery = bigramModel[_tquery]
	if DEBUG:
		print("QUERY:", tquery, "(aggregated token: ", list(set(tquery) - set(_tquery)), ")")

	# Computing query similarity with documents from corpus
	queryVec = (np.sum(np.array([getEmbedding(model, x) for x in tquery], dtype=float), axis=0))
	for doc, vec in docVec.items():
		if vec.ndim:
			res[doc] = getSimilarity(queryVec, vec, model)
	return dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:topN])

def getBestResultsDocEmbeddings(queryStr, topN, model):
	query = queryStr.split(" ")
	# res = defaultdict(float)

	# Tokenizing and stemming query
	tquery = set()
	for word in query:
		word = stemmer.stem(word.lower())
		if word not in stopWords and len(word) > 2:
			tquery.add(word)
	
	# Computing query similarity with documents from corpus
	queryVec = model.infer_vector(tquery)
	res = model.dv.most_similar(queryVec, topn=topN)
	return {doc: score for (doc, score) in res}
	
	# return dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:topN])

# Sorts a list of results according to their pageRank
def rankResults(results):
	return dict(sorted(results.items(), key=lambda item: pageRanks.index(item[0])))

def printResults(rankedResults):
	for idx, page in enumerate(rankedResults):
		if DEBUG:
			print(str(idx + 1) + ". " + page + " (sim-score: " + str(rankedResults[page]) + ", rank-score: " + str(pageRankDict[page]) + ")")
		else:
			print(str(idx + 1) + ". " + page)


if __name__ == '__main__':
	# Running options
	DEBUG = False # prints scores

	with open("tfidf.dict",'rb') as f:
		tfidf = pickle.load(f)

	with open("tfidf-norm.dict",'rb') as f:
		tfidfNorm = pickle.load(f)
		
	with open("tokInfo.dict",'rb') as f:
		tokInfo = pickle.load(f)

	with open("pageRank.dict",'rb') as f:
		pageRankDict = pickle.load(f)
		sortedPageRankDict = dict(sorted(pageRankDict.items(), key=lambda item: item[1], reverse=True))
		pageRanks = list(sortedPageRankDict.keys())

	with open("docVecCBOW.dict",'rb') as f:
		docVecCBOW = pickle.load(f)

	bigramModel = Phraser.load("bigram.model")
	cbowModel = KeyedVectors.load("cbow.model")
	DMdocModel = Doc2Vec.load("dm.model")

	query = sys.argv[1] if len(sys.argv) > 1 else "darwin"	# Query
	top = int(sys.argv[2]) if len(sys.argv) > 2 else 5		# Number of results to show
	
	print()
	print("###############################")
	print("Results with tf-idf model")
	print("###############################")
	print()
	t1 = time()

	print("Results for", query, "\n===========")
	results = getBestResults(query, top, tfidf)
	printResults(results)

	print("\nResults after normalization for", query, "\n===========")
	results = getBestResults(query, top, tfidfNorm)
	printResults(results)

	print("\nResults after ranking for", query, "\n===========")
	results = rankResults(results)
	printResults(results)

	print()
	print("###############################")
	print("Results with word2vec embeddings")
	print("###############################")
	print()

	print("Results with CBOW for", query, "\n===========")
	results = getBestResultsWordEmbeddings(query, top, cbowModel, docVecCBOW, bigramModel)
	printResults(results)

	print("\nResults with CBOW after ranking for", query, "\n===========")
	results = rankResults(results)
	printResults(results)
	
	print()
	print("###############################")
	print("Results with doc2vec embeddings")
	print("###############################")
	print()

	print("Results with Distributed Memory for", query, "\n===========")
	results = getBestResultsDocEmbeddings(query, top, DMdocModel)
	printResults(results)

	print("\nResults with Distributed Memory after ranking for", query, "\n===========")
	results = rankResults(results)
	printResults(results)

	t2 = time()
	print()
	print('Done in:', t2 - t1, "seconds")
