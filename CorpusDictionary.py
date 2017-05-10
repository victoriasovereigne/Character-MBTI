import json
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import *
import gensim
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np 
import os
import random
from gensim import models

import gensim
import nltk
from nltk.corpus import wordnet as wn
from gensim import corpora
import math

from nltk.corpus import verbnet as vn 
from nltk import FreqDist
from nltk.tokenize import word_tokenize

from Character import *

class CorpusDictionary:
	def __init__(self, json_folder, lemmatized=True, dictExists=False):
		self.json_folder = json_folder
		self.corpora = {'agent':None, 'patient':None, 'mod':None, 'poss':None, 'all':None}
		self.character_list = {}

		json_files = os.listdir(json_folder)
		lmt = WordNetLemmatizer()

		agent = []
		patient = []
		mod = []
		poss = []

		for afile in json_files:
			# print(afile)
			name = afile[:-5]
			c = Character(name)
			afile = json_folder + '/' + afile

			try:
				f = open(afile, 'r')
				data = json.load(f)
				c.E = float(data['extroversion'])
				c.A = float(data['agreeableness'])
				c.C = float(data['conscientiousness'])
				c.S = float(data['stability'])
				c.O = float(data['openness'])

				c.zE = float(data['z_extroversion'])
				c.zA = float(data['z_agreeableness'])
				c.zC = float(data['z_conscientiousness'])
				c.zS = float(data['z_stability'])
				c.zO = float(data['z_openness'])

				c.gender = int(data['gender'])
				c.salience = int(data['salience'])
				c.valence = int(data['valence'])

				for word in data['agent']:
					if lemmatized:
						c.persona['agent'].append(word[1].lower())
					else:
						c.persona['agent'].append(word[0].lower())

				for word in data['patient']:
					if lemmatized:
						c.persona['patient'].append(word[1].lower())
					else:
						c.persona['patient'].append(word[0].lower())

				for word in data['mod']:
					# print('===============================')
					# print(word[1])
					# self.is_person(word[1])
					if lemmatized:
						c.persona['mod'].append(word[1].lower())
					else:
						c.persona['mod'].append(word[0].lower())

				for word in data['poss']:
					# print('===============================')
					# print(word[1])
					# self.is_person(word[1])
					if lemmatized:
						c.persona['poss'].append(word[1].lower())
					else:
						c.persona['poss'].append(word[0].lower())

				agent.append(c.persona['agent'])
				patient.append(c.persona['patient'])
				mod.append(c.persona['mod'])
				poss.append(c.persona['poss'])

				self.character_list[name] = c

			except Exception as e:
				print(afile, e)

		aall = agent + patient + mod + poss
		# print([item for sublist in aall for item in sublist])
		fd = FreqDist([item for sublist in poss for item in sublist])
		for f in fd.most_common(1000):
			print('===============================')
			print(f)
			# print(vn.classids(lemma=f[0]))
			self.is_person(f[0])

		self.corpora['agent'] = Dictionary(agent)
		self.corpora['patient'] = Dictionary(patient)
		self.corpora['mod'] = Dictionary(mod)
		self.corpora['poss'] = Dictionary(poss)
		self.corpora['all'] = Dictionary(agent+patient+mod+poss)

		# self.corpora['agent'].filter_extremes(no_below=3, no_above=0.9)
		# self.corpora['patient'].filter_extremes(no_below=3, no_above=0.9)
		# self.corpora['mod'].filter_extremes(no_below=3, no_above=0.9)
		# self.corpora['poss'].filter_extremes(no_below=3, no_above=0.9)
		# self.corpora['all'].filter_extremes(no_below=3, no_above=0.9)

	# =========================================================
	# Converting the characters' personas into vectors
	# =========================================================
	def convert_character_to_vector(self):
		for character in self.character_list.keys():
			# print(character)
			char = self.character_list[character]
			myall = []

			for mytype in ['agent', 'patient', 'mod', 'poss']:
				persona = char.persona[mytype]
				myall.extend(persona)

				char.vector[mytype] = self.corpora[mytype].doc2bow(persona)
				# print(mytype)
				# print(self.character_list[character].vector[mytype])

			char.vector['all'] = self.corpora['all'].doc2bow(myall)


	# save the dictionary into files
	def save(self):
		self.corpora['agent'].save('agent.dict')
		self.corpora['patient'].save('patient.dict')
		self.corpora['poss'].save('poss.dict')
		self.corpora['mod'].save('mod.dict')
		self.corpora['all'].save('all.dict')

	def is_person(self, word):
		synsets = wn.synsets(word)

		for i in range(1, len(synsets)+1):
			x = ''
			if i < 10:
				x = '0'+str(i)
			else:
				x = str(i)
			try:
				n = word + '.n.' + x
				obj = wn.synset(n)
				hyper = lambda s: s.hypernyms()
				list_hyper = list(obj.closure(hyper))
				print(list_hyper)

				for elmt in list_hyper:
					if 'person.n.01' == elmt.name():
						return True
			except:
				continue

		return False

# print(vn.classids(lemma='take'))
# print(vn.classids(lemma='speak'))
# print(vn.classids(lemma='say'))