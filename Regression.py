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
					word = word.lower()
					if lemmatized:
						word = lmt.lemmatize(word, 'v')
					c.persona['agent'].append(word)

				for word in data['patient']:
					word = word.lower()
					if lemmatized:
						word = lmt.lemmatize(word, 'v')
					c.persona['patient'].append(word)

				for word in data['mod']:
					word = word.lower()
					if lemmatized:
						word = lmt.lemmatize(word, 'v')
					c.persona['mod'].append(word)

				for word in data['poss']:
					word = word.lower()
					if lemmatized:
						word = lmt.lemmatize(word, 'v')
					c.persona['poss'].append(word)

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

	def convert_character_to_vector(self):
		for character in self.character_list.keys():
			# print(character)
			char = self.character_list[character]
			all = []

			for type in ['agent', 'patient', 'mod', 'poss']:
				persona = char.persona[type]
				all.extend(persona)

				char.vector[type] = self.corpora[type].doc2bow(persona)
				
				# print(type)
				# print(self.character_list[character].vector[type])

			char.vector['all'] = self.corpora['all'].doc2bow(all)

	def save(self):
		self.corpora['agent'].save('agent.dict')
		self.corpora['patient'].save('patient.dict')
		self.corpora['poss'].save('poss.dict')
		self.corpora['mod'].save('mod.dict')
		self.corpora['all'].save('all.dict')


class Regression:
	def __init__(self, corpus_dict, train=0.8, validation=0.1):
		self.corpus_dict = corpus_dict

		char_list = self.corpus_dict.character_list
		keys = sorted(char_list.keys())

		self.X_train_agent = []
		self.X_train_patient = []
		self.X_train_poss = []
		self.X_train_mod = []
		self.X_train_all = []
		
		self.y_train_E = []
		self.y_train_A = []
		self.y_train_C = []
		self.y_train_S = []
		self.y_train_O = []

		self.y_train_zE = []
		self.y_train_zA = []
		self.y_train_zC = []
		self.y_train_zS = []
		self.y_train_zO = []

		train_num = int(0.9 * len(keys))

		for key in keys[:train_num]:
			c = char_list[key]
			self.X_train_agent.append(self.doc2bow_to_sparse_vector(c.vector['all'], len(self.corpus_dict.corpora['all'])))
			self.y_train_E.append(c.zE)

		self.X_test_agent = []
		self.X_test_patient = []
		self.X_test_poss = []
		self.X_test_mod = []
		self.X_test_all = []
		
		self.y_test_E = []
		self.y_test_A = []
		self.y_test_C = []
		self.y_test_S = []
		self.y_test_O = []

		self.y_test_zE = []
		self.y_test_zA = []
		self.y_test_zC = []
		self.y_test_zS = []
		self.y_test_zO = []

		for key in keys[train_num:]:
			c = char_list[key]
			self.X_test_agent.append(self.doc2bow_to_sparse_vector(c.vector['all'], len(self.corpus_dict.corpora['all'])))
			self.y_test_E.append(c.zE)

	def create_linear_model(self):
		self.svr = SVR(C=1.0, epsilon=0.2)
		self.svr.fit(self.X_train_agent, self.y_train_E)
		newY = self.svr.predict(self.X_test_agent)
		print(newY)
		print(self.y_test_E)

		print(mean_squared_error(self.y_test_E, newY))

	def doc2bow_to_sparse_vector(self, d, col):
		ans = [0] * col

		for (a, b) in d:
			ans[a] = b

		return np.array(ans)

cd = CorpusDictionary('character_json')
cd.convert_character_to_vector()

r = Regression(cd)
r.create_linear_model()