import json
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import *
import gensim
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np 
import os
import random
from gensim import models
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

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

		# train_num = int(0.9 * len(keys))
		X = []
		y = []
		# print("============training==============")
		for key in keys:
			c = char_list[key]
			# print(c.name, c.gender, c.salience, c.valence)
			whatev = self.doc2bow_to_sparse_vector(c.vector['all'], len(self.corpus_dict.corpora['all']), gender=c.gender, salience=c.salience, valence=c.valence)
			X.append(whatev)
			y.append(c.E)
			# print(whatev)
			# self.X_train_agent.append(whatev)
			# self.y_train_E.append(c.zE)

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

		self.X_train_agent, self.X_test_agent, self.y_train_E, self.y_test_E = train_test_split(X, 
			y, test_size=0.1, random_state=0)

		# print("============testing==============")
		# for key in keys[train_num:]:
		# 	c = char_list[key]
		# 	# print(c.name, c.gender, c.salience, c.valence)
		# 	whatev = self.doc2bow_to_sparse_vector(c.vector['all'], len(self.corpus_dict.corpora['all']), gender=c.gender, salience=c.salience, valence=c.valence)
		# 	# print(whatev)
		# 	self.X_test_agent.append(whatev)

		# 	self.y_test_E.append(c.zE)

	def create_linear_model(self):
		self.svr = SVR(C=1, epsilon=0.2)
		self.svr.fit(self.X_train_agent, self.y_train_E)
		# print(self.svr.get_params())
		newY = self.svr.predict(self.X_test_agent)

		print(newY)
		print(self.y_test_E)

		print(mean_squared_error(self.y_test_E, newY))

		self.lin_reg = LinearRegression()
		self.lin_reg.fit(self.X_train_agent, self.y_train_E) # , sample_weight=[3]*len(self.y_train_E))
		newY = self.lin_reg.predict(self.X_test_agent)

		tokens = {k:v for (k,v) in self.corpus_dict.corpora['all'].items()}
		# print(tokens)
		# print(self.lin_reg.coef_)

		for i,coef in enumerate(self.lin_reg.coef_):
			print(coef, tokens.get(i, ''))

		print(len(self.lin_reg.coef_))

		for i in range(len(newY)):
			print(newY[i], self.y_test_E[i])
		# print(newY)
		# print(self.y_test_E)

		# predicted = cross_val_predict(self.lin_reg, self.X_train_agent, self.y_train_E, cv=10)
		# print(predicted)
		# print(mean_squared_error(predicted, self.y_train_E))
		print(mean_squared_error(self.y_test_E, newY))
		# print(self.lin_reg.score(self.X_test_agent, self.y_test_E))
		print(pearsonr(self.y_test_E, newY))

	def doc2bow_to_sparse_vector(self, d, col, sense=[], gender=0, salience=0, valence=0):
		length =  col + len(sense) + 3
		ans = [0] * length
		ans[length-1] = valence
		ans[length-2] = salience
		ans[length-3] = gender

		total = 0
		for (a, b) in d:
			total += b
		# print(total)
		# ans[length-4] = total
 
		for (a, b) in d:
			ans[a] = b / total

		return np.array(ans)

cd = CorpusDictionary('character_json1')
cd.convert_character_to_vector()

r = Regression(cd)
r.create_linear_model()