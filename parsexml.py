import sys
import os
import xml.etree.ElementTree
import re
import pickle
import glob
from math import log10
from itertools import chain
from time import time

from copy import deepcopy

from collections import Counter
from os.path import exists
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import multiprocessing
cores = multiprocessing.cpu_count()

import json
class SetEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, set):
			return list(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

# Gets a word embedding according to a model (KeyedVector instance)
def getEmbedding(model, word):
	return model.get_vector(word) if word in model.key_to_index else np.zeros(model.vector_size)

if __name__ == '__main__':
	# Running options
	DEBUG = False # dumps main dictionaries to JSON files

	xmlFiles = list(chain(*[ glob.glob(globName)  for globName in sys.argv[1:] ]))
	docs = dict()

	# Cleaning debug files
	if not DEBUG:
		dir = "./"
		for f in os.listdir(dir):
			if re.search("(_*.json)", f):
				os.remove(os.path.join(dir, f))

	##############################
	print("Parsing XML...")
	##############################
	for xmlFile in xmlFiles:
		pages = xml.etree.ElementTree.parse(xmlFile).getroot()

		for page in pages.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
			titles = page.findall("{http://www.mediawiki.org/xml/export-0.10/}title")
			revisions = page.findall("{http://www.mediawiki.org/xml/export-0.10/}revision")
		
			if titles and revisions:
				revision = revisions[0] # last revision
				contents = revision.findall("{http://www.mediawiki.org/xml/export-0.10/}text")
				if contents:
					docs[titles[0].text] = contents[0].text 

	# Some regEx for parsing
	# https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout
	# extLinkRe = '({{)([^{}]+)(}})\n?'
	extLinkRe2 = '(\[[^\]]+\])'
	extLinkRe3 = '({{[^{}]+)({{[^{}]+}})?([^{}]*}})\n?'
	refRe = '(<ref[^\/]+\/>)|(<ref[^>]*>)[^<]+|(<ref[^>]*>[^>])|(<\/ref>)|(\'\')'
	linkRe = '\[\[([^\]\|]+)(\|[^\]\|]+)?\]\]'
	falseLinkRe = '{{(?!cite)[^}]+\|([^\|}]+)}}|{{([^\|}]+)}}'
	titleRe = ' ?(=+) ?'
	categoryRe = '\n?(category:.+)\n?'
	fileRe = '(\[\[file:[^\]]+\]\])'
	classRe = '{\| ?class=".+"\n?([^}]+)}'
	imgSizeRe = '([0-9]+x[0-9]+px)'
	htmlCommentRe = '(<!--.+-->)'
	htmlExtLinkRe = '(<[^>]+>)'

	removeLinkRe = "\[\[[^\]]+\|([^\|\]]+)\]\]"
	removeLink2Re =  "\[\[([^\|\]]+)\]\]"
	wordRe = "[a-zA-Z\-]+"
	# stopWords = ["-"]
	stopWords = set(stopwords.words('english'))

	##############################
	print("Extracting links, transforming links in text, tokenizing, stemming, building corpus and filling doc-tok/tok-doc matrices...")
	##############################
	t1 = time()

	links = dict()
	doctok = dict()
	tokdoc = dict()
	corpus = list()
	stemmer = PorterStemmer() # Most used stemmer for information retrieval in english, based on suffixes

	for idx, doc in enumerate(docs):
		# Progress status
		if idx % (len(docs) // 20) == 0:
			print("Progress", int(idx * 100 / len(docs)),  "%")
		# Looking for links to other wiki pages
		links[doc] = list()
		for link in re.finditer(linkRe, docs[doc]):
			target = link.group(1).split('|')[0]
			if target in docs.keys():
				# print(doc, " --> ", target)
				links[doc] += [target]

		docs[doc] = re.sub(refRe, '', docs[doc])

		if DEBUG:
			if(idx == 4) :
				with open("_origin.txt", "w") as origin :
					origin.write(docs[doc])
			
		# Transforming links to text and cleaning from other tags
		# docs[doc] = re.sub(extLinkRe, '', docs[doc])
		docs[doc] = docs[doc].lower()
		docs[doc] = re.sub(extLinkRe3, '', docs[doc])
		docs[doc] = re.sub(falseLinkRe, r'\g<1>', docs[doc])
		docs[doc] = re.sub(classRe, r'\g<1>', docs[doc])
		docs[doc] = re.sub(removeLink2Re, r'\g<1>', docs[doc])
		docs[doc] = re.sub(fileRe, '', docs[doc])
		docs[doc] = re.sub(removeLinkRe, r'\g<1>', docs[doc])
		docs[doc] = re.sub(categoryRe, '', docs[doc])
		docs[doc] = re.sub(titleRe, '', docs[doc])
		docs[doc] = re.sub(extLinkRe2, '', docs[doc])
		docs[doc] = re.sub(imgSizeRe, '', docs[doc])
		docs[doc] = re.sub(htmlCommentRe, '', docs[doc])
		docs[doc] = re.sub(htmlExtLinkRe, '', docs[doc])
		docs[doc] = docs[doc].replace('\n', ' ')
		for specialChar in ['.', '/', '\\', '(', ')', '{', '}', '*', '-', '~', '\'', '"', '_']:
			docs[doc] = docs[doc].replace(specialChar, '')
		for title in ["further reading", "references", "external links", "see also", "__notoc__"]:
			docs[doc] = docs[doc].replace(title, '')

		### NAIVE WAY TO COMPUTE TOKDOC FROM DOCTOK		
		# Fill the doc-tok matrix - sparse index
		# doctok[doc] = list()
		# for wordre in re.finditer(wordRe, cleanDoc):
		# 	word = wordre.group(0).lower()
		# 	word = stemmer.stem(word) # Stemming

		# 	if word not in stopWords and len(word) > 2:
		# 		doctok[doc] += [word]
		
		# Fill the tok-doc matrix from the doc-tok matrix - reversed sparse index
		# for doc in doctok:
		# 	for tok in doctok[doc]:
		# 		if tok not in tokdoc.keys():
		# 			tokdoc[tok] = set()
		# 		tokdoc[tok].add(doc)

		# Fill the doc-tok and tok-doc matrices directly - sparse and reversed sparse indexes
		doctok[doc] = list()
		for wordre in re.finditer(wordRe, docs[doc]):
			word = wordre.group(0)
			token = stemmer.stem(word) # Stemming

			if token not in stopWords and len(token) > 2:
				docs[doc] = docs[doc].replace(word, token)
				doctok[doc] += [token]
				if token not in tokdoc.keys():
					tokdoc[token] = set()
				tokdoc[token].add(doc)

		corpus += [docs[doc].split(" ")]

		if DEBUG:
			if(idx == 4):
				with open("_cleaned.txt", "w") as cleaned :
					cleaned.write(docs[doc])

	if DEBUG:
		with open("_doctok.json", "w") as f :
			f.write(json.dumps(doctok, indent=4, cls=SetEncoder))
		with open("_tokdoc.json", "w") as f :
			f.write(json.dumps(tokdoc, indent=4, cls=SetEncoder))
		with open("_corpus.json", "w") as f :
			f.write(json.dumps(corpus, indent=4, cls=SetEncoder))

	print("done.")

	##############################
	print("Building tf & idf...")
	##############################
	docList = doctok.keys()
	Ndocs = len(docList)

	# Computing tc (term-count) tf (term-frequency / tc normailized with doc's size)
	tc = dict() # tc[doc][token] contains the count of the token "token" in document "doc"
	tf = dict() # tf[doc][token] contains the relative frequency of the token "token" in document "doc"
	for doc in docList:
		N = len(doctok[doc])
		tc[doc] = dict(Counter(doctok[doc])) # Counting occurrences
		tf[doc] = deepcopy(tc[doc])
		for tok in tf[doc]:
			tf[doc][tok] /= N # Get relative frequency

	if DEBUG:
		with open("_tc.json", "w") as f :
			f.write(json.dumps(tc, indent=4, cls=SetEncoder))
		with open("_tf.json", "w") as f :
			f.write(json.dumps(tf, indent=4, cls=SetEncoder))
		
	# Computing idf (inversed document frequency)
	tokInfo = dict() # tokInfo[tok] contains the information in bits of the token
	for token in tokdoc:
		tokInfo[token] = log10(Ndocs / float(len(tokdoc[token])))

	if DEBUG:
		with open("_idf.json", "w") as f :
			f.write(json.dumps(tokInfo, indent=4, cls=SetEncoder))

	print("done.")

	###############################
	print("Creating tf-idf...")
	###############################
	tfidf = dict() # reversed sparse index format, using term-count
	tfidfNorm = dict() # reversed sparse index format, using term-frequency
	for tok in tokdoc:
		tfidf[tok] = dict()
		tfidfNorm[tok] = dict()
		for doc in tokdoc[tok]:
			tfidf[tok][doc] = tc[doc][tok] * tokInfo[tok]
			tfidfNorm[tok][doc] = tf[doc][tok] * tokInfo[tok]

	if DEBUG:
		with open("_tfidf.json", "w") as f :
			f.write(json.dumps(tfidf, indent=4, cls=SetEncoder))
		with open("_tfidfNorm.json", "w") as f :
			f.write(json.dumps(tfidfNorm, indent=4, cls=SetEncoder))

	print("done.")

	t2 = time()
	print('Done in:', t2 - t1, "seconds") # ~50sec for matrices directly constructed, >500sec in the naive way

	###############################
	print("Creating Phrases and CBOW word2vec model...")
	###############################
	t1 = time()

	# Creating Phrases model to find common phrases from corpus (bigrams)
	phrases = Phrases(corpus, min_count=3)
	bigram = Phraser(phrases)
	corpus_sentences = bigram[corpus]

	if DEBUG:
		# print(list(set(corpus_sentences[0]) - set(corpus[0]))) # Newly created bigram tokens
		with open("_corpus-sentences.json", "w") as f :
			f.write(json.dumps(list(corpus_sentences), indent=4, cls=SetEncoder))

	# Creating Continuous Bag of Words (CBOW) model to predict the current word given context words
	# Keeping only word vectors because we don't need to continue model's training
	_cbowModel = gensim.models.Word2Vec(vector_size=300, window=5, min_count=2, sample=6e-3, sg=0, workers=cores)
	_cbowModel.build_vocab(corpus_sentences)
	_cbowModel.train(corpus_sentences, total_examples=_cbowModel.corpus_count, epochs=30, report_delay=1)
	cbowModel = _cbowModel.wv
	del _cbowModel # Let's free some RAM

	# Computing average vector for each document
	docVecCBOW = dict()
	for idx, doc in enumerate(docList):
		docVecCBOW[doc] = np.sum(np.array([getEmbedding(cbowModel, x) for x in corpus_sentences[idx]]), axis=0)

	if DEBUG:
		with open("_docVecCBOW.json", "w") as f :
			f.write(json.dumps(docVecCBOW, indent=4, cls=SetEncoder))

	t2 = time()
	print('Done in:', t2 - t1, "seconds")

	###############################
	print("Creating Distributed Memory doc2vec model...")
	###############################
	t1 = time()

	# Tag data according to doc names and doc contents
	taggedData = [TaggedDocument(words=docContent.split(" "), tags=[doc]) for doc, docContent in docs.items()]
	# Creating Distributed Memory model to preserves the word order in a document, unlike Bag of Words algorithms
	DMdocModel = Doc2Vec(taggedData, vector_size=100, window=10, min_count=2, epochs=30, dm=1, workers=cores)

	t2 = time()
	print('Done in:', t2 - t1, "seconds")

	###############################
	print("Saving links, matrices, tf-idf and document vectors as pickle objects...")
	###############################
	with open("links.dict",'wb') as fileout:
		pickle.dump(links, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("doctok.dict",'wb') as fileout:
		pickle.dump(doctok, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("tokdoc.dict",'wb') as fileout:
		pickle.dump(tokdoc, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("tfidf.dict",'wb') as fileout:
		pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("tfidf-norm.dict",'wb') as fileout:
		pickle.dump(tfidfNorm, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("tokInfo.dict",'wb') as fileout:
		pickle.dump(tokInfo, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	with open("docVecCBOW.dict",'wb') as fileout:
		pickle.dump(docVecCBOW, fileout, protocol=pickle.HIGHEST_PROTOCOL)

	###############################
	print("Saving word2vec & doc2vec models...")
	###############################
	bigram.save("bigram.model")
	cbowModel.save("cbow.model")
	DMdocModel.save("dm.model")

	print("done.")
