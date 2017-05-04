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
from CorpusDictionary import *

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
		# print(newY)
		# print(self.y_test_E)

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