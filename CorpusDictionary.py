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

				for word in data['agent']:
					if lemmatized:
						c.persona['agent'].append(word[1])
					else:
						c.persona['agent'].append(word[0])

				for word in data['patient']:
					if lemmatized:
						c.persona['patient'].append(word[1])
					else:
						c.persona['patient'].append(word[0])

				for word in data['mod']:
					if lemmatized:
						c.persona['mod'].append(word)
					else:
						c.persona['mod'].append(word[0])

				for word in data['poss']:
					if lemmatized:
						c.persona['poss'].append(word)
					else:
						c.persona['poss'].append(word[0])

				agent.append(c.persona['agent'])
				patient.append(c.persona['patient'])
				mod.append(c.persona['mod'])
				poss.append(c.persona['agent'])

				self.character_list[name] = c

			except Exception as e:
				print(afile, e)

		self.corpora['agent'] = Dictionary(agent)
		self.corpora['patient'] = Dictionary(patient)
		self.corpora['mod'] = Dictionary(mod)
		self.corpora['poss'] = Dictionary(poss)
		self.corpora['all'] = Dictionary(agent+patient+mod+poss)

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
				
				print(self.character_list[character].vector[mytype])

			char.vector['all'] = self.corpora['all'].doc2bow(myall)


	# save the dictionary into files
	def save(self):
		self.corpora['agent'].save('agent.dict')
		self.corpora['patient'].save('patient.dict')
		self.corpora['poss'].save('poss.dict')
		self.corpora['mod'].save('mod.dict')
		self.corpora['all'].save('all.dict')
