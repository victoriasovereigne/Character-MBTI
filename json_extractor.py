import json
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import *
import gensim
from sklearn import svm
import numpy as np 
import os
import random
from gensim import models

from Character import *

class Character_Extractor:
	def __init__(self, gold_standard):
		self.gold_standard = gold_standard
		self.book_characters = {}

		f = open(gold_standard, 'r')
		lines = f.readlines()

		for line in lines:
			data = line.split(',')
			c = Character(data[0])
			character_name = data[0]
			book = data[1]
			names_in_book = data[2].split('/')
			other_name = s[3]
			sex = s[4]
			_ae = s[8]
			_aa = s[9]
			_ac = s[10]
			_as = s[11]
			_ao = s[12]

			if book not in self.book_characters.keys():
				self.book_characters[book] = 



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

# d = extract_character('book.id/pride_and_prejudice.book', 'Elizabeth')
# x1 = extract_property(d, 'agent')
# print(x1)
# x1 = extract_property(d, 'mod')
# print(x1)
# x1 = extract_property(d, 'poss')
# print(x1)
# x1 = extract_property(d, 'patient')
# print(x1)

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

def create_dictionary(gold_file, book_folder, dictionary = None, with_tfidf=False):
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
			ext_prop3 = extract_property(ext_char, 'poss')
			ext_prop4 = extract_property(ext_char, 'patient')

			total = ext_prop + ext_prop2 + ext_prop3 + ext_prop4

			d.append(total)
			
			chara_dict[name] = total
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
	for i in range(10):
		cutoff = int(len(chara_dict.keys())*0.8)
		keys = list(chara_dict.keys())
		random.shuffle(keys)
		training_set = keys[:cutoff]
		testing_set = keys[cutoff:]

		X = []
		y = []
		corpus = []
		chara_db = {}
		for name in training_set:
			# print(name)
			chara_db[name] = dictionary.doc2bow(chara_dict[name]) # corpus
			corpus.append(chara_db[name])

			# for (a, b) in db:
			# 	print(dictionary.get(a), b)

		tfidf = models.TfidfModel(corpus)

		for name in training_set:
			db = chara_db[name]

			if with_tfidf:
				db = tfidf[chara_db[name]]
				
			db_sparse = doc2bow_to_sparse_vector(db, len(dictionary))
			ctype = chara_type[name]
			X.append(db_sparse)
			# y.append(ctype)

		# clf = svm.SVC()
		# clf.fit(X, y)

		# accuracy = 0
		# tighter_accuracy = 0

		# for name in testing_set:
		# 	db = dictionary.doc2bow(chara_dict[name])
		# 	db_sparse = doc2bow_to_sparse_vector(db, len(dictionary))
		# 	prediction = clf.predict(db_sparse)[0]
		# 	actual = chara_type[name]

		# 	print(name, "predicted:", prediction, ", actual:", actual)
			
		# 	if actual == prediction:
		# 		accuracy += 1
			
		# 	if actual[0] == prediction[0]:
		# 		tighter_accuracy += 0.25
		# 	if actual[1] == prediction[1]:
		# 		tighter_accuracy += 0.25
		# 	if actual[2] == prediction[2]:
		# 		tighter_accuracy += 0.25
		# 	if actual[3] == prediction[3]:
		# 		tighter_accuracy += 0.25

		# print("Accuracy:", accuracy/len(testing_set))
		# print("Tighter Accuracy:", tighter_accuracy/len(testing_set))


# create_dictionary('mini_gold_standard.csv', 'book.id')
# dictionary = Dictionary.load('dictionary_both.ser')
# print(dictionary.token2id)


def create_dictionary2(gold_file='mini-goldstandard.csv', book_folder='book.id', save_file='dict.ser'):
	d = []
	chara_dict = {}
	gf = open(gold_file, 'r')
	lines = gf.readlines()

	for line in lines:
		s = line.split(',')
		name = s[0]
		book = s[1]
		name_in_book = s[2]
		other_name = s[3]
		sex = s[4]
		_ae = s[8]
		_aa = s[9]
		_ac = s[10]
		_as = s[11]
		_ao = s[12]
		
		book_file = book_folder + '/' + book

		try:
			ext_char = extract_character(book_file, name_in_book)
			ext_prop = extract_property(ext_char, 'agent')
			ext_prop2 = extract_property(ext_char, 'mod')
			ext_prop3 = extract_property(ext_char, 'poss')
			ext_prop4 = extract_property(ext_char, 'patient')

			total = ext_prop + ext_prop2 + ext_prop3 + ext_prop4

			if other_name != '-':
				ext_char = extract_character(book_file, other_name)
				ext_prop = extract_property(ext_char, 'agent')
				ext_prop2 = extract_property(ext_char, 'mod')
				ext_prop3 = extract_property(ext_char, 'poss')
				ext_prop4 = extract_property(ext_char, 'patient')

				total += ext_prop + ext_prop2 + ext_prop3 + ext_prop4

			d.append(total)
			
			chara_dict[name] = total
		except Exception as e:
			print(name, e)

	dictionary = Dictionary(d)
	dictionary.save(save_file)
	return dictionary, chara_dict


	