import json
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import *
import gensim
from sklearn import svm
import numpy as np 
import os

def extract_character(filename, character_name):
	data_file = open(filename, 'r')
	data = json.load(data_file)

	answer = {}
	tmp = 0

	for i in range(len(data['characters'])):
		names = data['characters'][i]['names']

		for name in names:
			if character_name == name['n'] and tmp < name['c']:
				answer = data['characters'][i]
				tmp = name['c']

	return answer

# pprint(extract_character('austen.book.id', 'Elinor'))

def extract_property(data, prop):
	answer = []
	lmt = WordNetLemmatizer()

	for d in data[prop]:
		answer.append(lmt.lemmatize(d['w'].lower(), 'v'))

	return answer

# d = extract_character('sense_and_sensibilities.book', 'Marianne')
# x1 = extract_property(d, 'agent')
# # # print(x1)

# d = extract_character('sense_and_sensibilities.book', 'Elinor')
# x2 = extract_property(d, 'agent')
# # # print(x2)

# d = extract_character('sense_and_sensibilities.book', 'Edward')
# x3 = extract_property(d, 'agent')
# # # print(x3)

# d = extract_character('sense_and_sensibilities.book', 'Willoughby')
# x4 = extract_property(d, 'agent')
# # # print(x4)

# dictionary = Dictionary([x1, x2, x3, x4])

# d1 = dictionary.doc2bow(x1)
# d2 = dictionary.doc2bow(x2)
# d3 = dictionary.doc2bow(x3)
# d4 = dictionary.doc2bow(x4)

# print(len(dictionary))

def doc2bow_to_sparse_vector(d, col):
	ans = [0] * col

	for (a, b) in d:
		ans[a] = b

	return ans

# print(doc2bow_to_sparse_vector(d1, 380))
# print(doc2bow_to_sparse_vector(d2, 380))
# print(doc2bow_to_sparse_vector(d3, 380))
# print(doc2bow_to_sparse_vector(d4, 380))

# v1 = doc2bow_to_sparse_vector(d1, 380)
# v2 = doc2bow_to_sparse_vector(d2, 380)
# v3 = doc2bow_to_sparse_vector(d3, 380)
# v4 = doc2bow_to_sparse_vector(d4, 380)

# marianne - infp
# elinor - istj
# edward - isfj
# willoughby - estp

# X = np.array([v1, v2, v3])
# y = np.array([0, 1, 2])

# clf = svm.SVC()
# clf.fit(X, y)

# print(clf.predict(v4))

def create_dictionary(gold_file, book_folder, dictionary = None):
	d = []
	gf = open(gold_file, 'r')
	lines = gf.readlines()

	chara_dict = {}
	chara_type = {}

	for line in lines:
		s = line.split(',')
		name = s[0]
		book = s[1]
		type = s[2].strip()
		
		book_file = book_folder + '/' + book

		try:
			ext_char = extract_character(book_file, name)
			ext_prop = extract_property(ext_char, 'agent')
			ext_prop2 = extract_property(ext_char, 'mod')
			d.append(ext_prop + ext_prop2)
			
			chara_dict[name] = ext_prop
			chara_type[name] = type
		except Exception as e:
			print(name, e)

	if dictionary is None:
		dictionary = Dictionary(d)
		dictionary.save('dictionary_both.ser')
	else:
		dictionary = Dictionary.load(dictionary)

	# doc2bow
	# chara_bow = {}
	cutoff = int(len(chara_dict.keys())*0.8)
	training_set = sorted(chara_dict.keys())[:cutoff]
	testing_set = sorted(chara_dict.keys())[cutoff:]

	X = []
	y = []
	for name in training_set:
		db = dictionary.doc2bow(chara_dict[name])
		db_sparse = doc2bow_to_sparse_vector(db, len(dictionary))
		ctype = chara_type[name]
		X.append(db_sparse)
		y.append(ctype)

	clf = svm.SVC()
	clf.fit(X, y)

	for name in testing_set:
		db = dictionary.doc2bow(chara_dict[name])
		db_sparse = doc2bow_to_sparse_vector(db, len(dictionary))
		print(name, clf.predict(db_sparse))

create_dictionary('mini_gold_standard.csv', 'book.id', 'dictionary_both.ser')
# dictionary = Dictionary.load('dictionary_both.ser')
# print(dictionary.token2id)