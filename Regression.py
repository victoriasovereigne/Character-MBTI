import json
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import *
import gensim
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
import numpy as np 
import os
import random
from gensim import models
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

from Character import *
from CorpusDictionary import *

class Regression:
	def __init__(self, corpus_dict, salient_only=False):
		self.corpus_dict = corpus_dict
		self.elasticnet = {}

		char_list = self.corpus_dict.character_list
		keys = sorted(char_list.keys())
		self.X_train = {}
		self.y_train = {}
		self.X_val = {}
		self.y_val = {}
		self.X_test = {}
		self.y_test = {}

		X = {}
		y = {}
		modes = ['agent', 'patient', 'mod', 'poss', 'all']
		ffm = ['E', 'A', 'C', 'S', 'O']

		for mode in modes:
			self.X_train[mode] = []
			self.X_val[mode] = []
			self.X_test[mode] = []
			X[mode] = []


		for f in ffm:
			self.y_train[f] = []
			self.y_train['z'+f] = []
			self.y_val[f] = []
			self.y_val['z'+f] = []
			self.y_test[f] = []
			self.y_test['z'+f] = []
			y[f] = []
			y['z'+f] = []

		num_data = 0

		# print("============training==============")
		for key in keys:
			c = char_list[key]

			if c.salience == 1:
				print(c.name, c.gender, c.salience, c.valence)

			for mode in modes:
				if (salient_only and c.salience == 1) or not salient_only:
					sparse = self.doc2bow_to_sparse_vector(c.vector[mode], len(self.corpus_dict.corpora[mode]), gender=c.gender, salience=c.salience, valence=c.valence)
					# print(sparse)
					X[mode].append(sparse)
					

			if (salient_only and c.salience == 1) or not salient_only:
				num_data +=1
				y['E'].append(c.E)
				y['A'].append(c.A)
				y['C'].append(c.C)
				y['S'].append(c.S)
				y['O'].append(c.O)

				y['zE'].append(c.zE)
				y['zA'].append(c.zA)
				y['zC'].append(c.zC)
				y['zS'].append(c.zS)
				y['zO'].append(c.zO)

		N = range(num_data)
		print("num data",num_data)

		nx, nx_test, ny, ny_test = train_test_split(N, N, test_size=0.2, random_state=0)
		nx_train, nx_val, ny_train, ny_val = train_test_split(nx, ny, test_size=0.25, random_state=0)

		# print(nx_train, nx_val, nx_test)
		# print(ny_train, ny_val, ny_test)

		for i in nx_train:
			for mode in modes:
				self.X_train[mode].append(X[mode][i])

			for f in ffm:
				self.y_train[f].append(y[f][i])
				self.y_train['z'+f].append(y['z'+f][i])

		for i in nx_val:
			for mode in modes:
				self.X_val[mode].append(X[mode][i])

			for f in ffm:
				self.y_val[f].append(y[f][i])
				self.y_val['z'+f].append(y['z'+f][i])

		for i in nx_test:
			for mode in modes:
				self.X_test[mode].append(X[mode][i])

			for f in ffm:
				self.y_test[f].append(y[f][i])
				self.y_test['z'+f].append(y['z'+f][i])

		# (230, 2789) (77, 2789) (77, 2789) --> agent
		# (230, 1275) (77, 1275) (77, 1275) --> patient
		# (230, 2207) (77, 2207) (77, 2207) --> mod
		# (230, 3633) (77, 3633) (77, 3633) --> poss
		# (230, 7178) (77, 7178) (77, 7178) --> all

		for m in modes:
			print(np.array(self.X_train[m]).shape, np.array(self.X_val[m]).shape, np.array(self.X_test[m]).shape)

		# for f in ffm:
		# 	print(np.array(self.y_train[f]).shape, np.array(self.y_val[f]).shape, np.array(self.y_test[f]).shape)

	def train_elasticnet_model(self, mode, ffm):
		X_train = np.array(self.X_train[mode])
		y_train = np.array(self.y_train[ffm])

		X_val = np.array(self.X_val[mode])
		y_val = np.array(self.y_val[ffm])

		l1ratios = np.linspace(0.1, 1, 10)

		mses = []
		alps = []
		verr = []

		for l1 in l1ratios:
			print(l1)
			enet = ElasticNetCV(l1_ratio=l1, cv=10)
			enet.fit(X_train, y_train)
			y_pred = enet.predict(X_val)
			mse = mean_squared_error(y_val, y_pred)
			v = enet.score(X_val, y_val)
			
			mses.append(mse)
			alps.append(enet.alpha_)
			verr.append(v)

		i_opt = np.argmin(mses)
		l1_opt = l1ratios[i_opt]
		alpha_opt = alps[i_opt]

		print("optimal l1", l1_opt)
		print("optimal alpha", alpha_opt)

		enet2 = ElasticNetCV(l1_ratio=l1_opt)
		enet2.fit(X_train, y_train)
		y_pred = enet2.predict(X_val)
		y_pred_train = enet2.predict(X_train)
		
		print("Training MSE", mean_squared_error(y_train, y_pred_train))
		print("Validation MSE", mean_squared_error(y_val, y_pred))

		print("Training Pearson R", pearsonr(y_train, y_pred_train))
		print("Validation Pearson R", pearsonr(y_val, y_pred))

		print("Training R2 score:", enet.score(X_train, y_train))
		print("Validation R2 score:", enet.score(X_val, y_val))

		# print(enet2.alpha_)

		self.elasticnet[(mode, ffm)] = enet2

		return self.elasticnet[(mode, ffm)]

	def test_elasticnet_model(self, mode, ffm):
		print("=================================")
		print(mode, ffm)
		print("=================================")
		X_test = np.array(self.X_test[mode])
		y_test = np.array(self.y_test[ffm])

		enet = self.elasticnet[(mode, ffm)]
		y_pred = enet.predict(X_test)

		tokens = {k:v for (k,v) in self.corpus_dict.corpora[mode].items()}
		pairs = []

		for i,coef in enumerate(enet.coef_):
			if i in tokens.keys():
				pairs.append((coef, tokens[i]))
			else:
				if i == (len(tokens) - 1):
					pairs.append((coef, 'valence'))
				elif i == (len(tokens) - 2):
					pairs.append((coef, 'salience'))
				elif i == (len(tokens) - 3):
					pairs.append((coef, 'gender'))	

		top10 = sorted(pairs, reverse=True)[:10]

		for (a, b) in top10:
			print(a, b)

		bottom10 = sorted(pairs)[:10]

		for (a, b) in bottom10:
			print(a, b)

		print("R2 score:", enet.score(X_test, y_test))
		print("MSE:", mean_squared_error(y_test, y_pred))
		print(pearsonr(y_test, y_pred))

		for i in range(len(y_test)):
			print(y_test[i], y_pred[i])
		print()


	def doc2bow_to_sparse_vector(self, d, col, sense=[], gender=0, salience=0, valence=0):
		length =  col + len(sense) + 3
		ans = [0] * length

		total = 0
		for (a, b) in d:
			total += b

		if total > 0:
 			for (a, b) in d:
 				ans[a] = b

		ans[length-1] = valence
		ans[length-2] = salience
		ans[length-3] = gender
		
		return np.array(ans)

# cd.get_hypernyms('friendly', 'a')

# ==============================================
# Exp mode: no below 10 documents
# Feature size:
# ==============================================
# (230, 617) (77, 617) (77, 617)
# (230, 233) (77, 233) (77, 233)
# (230, 151) (77, 151) (77, 151)
# (230, 455) (77, 455) (77, 455)
# (230, 1302) (77, 1302) (77, 1302)
# ==============================================

def main():
	print("=============================")
	print("To run: Regression.py < [param file]")
	print()
	print("Example: Regression.py < params.txt")
	print("=============================")

	json_folder = 'character_json1'
	word_freq = 'word_freq.json'
	pos_tag = 'pos_tag.json'
	abstraction = False
	sense = False
	nobelow = 20
	salient = False

	ffm = ''
	mode = ''

	for arg in sys.stdin:
		if arg.startswith('mode'):
			mode = arg.split('=')[1].strip()
		elif arg.startswith('ffm'):
			ffm = arg.split('=')[1].strip()
		elif arg.startswith('json_folder'):
			json_folder = arg.split('=')[1].strip()
		elif arg.startswith('word_freq'):
			word_freq = arg.split('=')[1].strip()
		elif arg.startswith('pos_tag'):
			pos_tag = arg.split('=')[1].strip()
		elif arg.startswith('abs'):
			if arg.split('=')[1].strip() == 'True':
				abstraction = True
		elif arg.startswith('sense'):
			if arg.split('=')[1].strip() == 'True':
				sense = True
		elif arg.startswith('nobelow'):
			nobelow = int(arg.split('=')[1].strip())
		elif arg.startswith('salient'):
			if arg.split('=')[1].strip() == 'True':
				salient = True

	# print(mode)
	cd = CorpusDictionary(json_folder, word_freq_file=word_freq, 
						pos_tag_file=pos_tag, abstraction_only=abs, get_sense=sense,
						no_below=nobelow, no_above=1)
	cd.convert_character_to_vector()

	r = Regression(cd, salient_only=True)

	r.train_elasticnet_model(mode, ffm)
	r.test_elasticnet_model(mode, ffm)

main()