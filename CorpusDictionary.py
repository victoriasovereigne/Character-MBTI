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
	def __init__(self, json_folder, lemmatized=True, num_filter=0, no_below=1, no_above=1, 
		word_freq_file=None, pos_tag_file=None, abstraction_only=False, get_sense=False):
		self.json_folder = json_folder
		self.corpora = {'agent':None, 'patient':None, 'mod':None, 'poss':None, 'all':None}
		self.character_list = {}

		json_files = os.listdir(json_folder)
		lmt = WordNetLemmatizer()

		agent = []
		patient = []
		mod = []
		poss = []
		self.modes = ['agent', 'patient', 'mod', 'poss']

		self.pos_tag = {}
		self.word_freq = {}

		self.initialize_pos_freq(json_folder, word_freq_file, pos_tag_file)

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

				for m in self.modes:
					for word in data[m]:
						w = word[1].lower()
						pos = self.pos_tag[w]

						if self.word_freq[w] >= num_filter:
							# -----------------------------
							# if we use the sense features
							# -----------------------------
							if get_sense:
								hypernyms = []
								if 'NN' in pos:
									hypernyms = self.get_hypernyms(w, 'n') 

								elif 'JJ' in pos:
									hypernyms = self.get_hypernyms(w, 'a') 

								elif 'VB' in pos:
									hypernyms = self.get_hypernyms(w, 'v') 
									print(w, pos, hypernyms)
								
								
								# the sense features from wordnet will be embedded in characters

							# -----------------------------
							# if we filter out physical entity
							# -----------------------------
							if abstraction_only and 'NN' in pos:
								hypernyms = self.get_hypernyms(w, 'n') 

								if len(hypernyms) > 0 and 'physical_entity.n.01' not in hypernyms:
									c.persona[m].append(w)
							else:
								c.persona[m].append(w)

				agent.append(c.persona['agent'])
				patient.append(c.persona['patient'])
				mod.append(c.persona['mod'])
				poss.append(c.persona['poss'])

				self.character_list[name] = c

			except Exception as e:
				print(afile, e)

		aall = agent + patient + mod + poss

		self.corpora['agent'] = Dictionary(agent)
		self.corpora['patient'] = Dictionary(patient)
		self.corpora['mod'] = Dictionary(mod)
		self.corpora['poss'] = Dictionary(poss)
		self.corpora['all'] = Dictionary(aall)

		self.corpora['agent'].filter_extremes(no_below=no_below, no_above=no_above)
		self.corpora['patient'].filter_extremes(no_below=no_below, no_above=no_above)
		self.corpora['mod'].filter_extremes(no_below=no_below, no_above=no_above)
		self.corpora['poss'].filter_extremes(no_below=no_below, no_above=no_above)
		self.corpora['all'].filter_extremes(no_below=no_below, no_above=no_above)

	# =========================================
	# create pos tag and count word frequencies
	# =========================================
	def initialize_pos_freq(self, json_folder, word_freq_file, pos_tag_file):
		json_files = os.listdir(json_folder)

		if word_freq_file is None and pos_tag_file is None:
			for afile in json_files:
				name = afile[:-5]
				afile = json_folder + '/' + afile

				try:
					f = open(afile, 'r')
					data = json.load(f)

					for m in self.modes:
						for word in data[m]:
							w = word[1].lower()
							self.pos_tag[w] = word[2]

							if w in self.word_freq.keys():
								self.word_freq[w] += 1
							else:
								self.word_freq[w] = 1
				except Exception as e:
					print(afile, e)

			p = open('pos_tag.json','w')
			json.dump(self.pos_tag, p)

			p = open('word_freq.json','w')
			json.dump(self.word_freq, p)
		else:
			with open(pos_tag_file, 'r') as f:
				self.pos_tag = json.load(f)
			with open(word_freq_file, 'r') as f:
				self.word_freq = json.load(f)
		# =========================================

	# =========================================================
	# Converting the characters' personas into vectors
	# =========================================================
	def convert_character_to_vector(self):
		for character in self.character_list.keys():
			# print(character)
			char = self.character_list[character]
			myall = []

			for mytype in self.modes:
				persona = char.persona[mytype]
				myall.extend(persona)

				char.vector[mytype] = self.corpora[mytype].doc2bow(persona)

			char.vector['all'] = self.corpora['all'].doc2bow(myall)

	# =========================================================
	# save the dictionary into files
	# =========================================================
	def save(self):
		self.corpora['agent'].save('agent.dict')
		self.corpora['patient'].save('patient.dict')
		self.corpora['poss'].save('poss.dict')
		self.corpora['mod'].save('mod.dict')
		self.corpora['all'].save('all.dict')

	def get_hypernyms(self, word, pos):
		synsets = wn.synsets(word)
		hypernyms = []

		if pos == 'a':
			return [s.name() for s in wn.synsets(word)]

		for i in range(1, len(synsets)+1):
			x = ''
			if i < 10:
				x = '0'+str(i)
			else:
				x = str(i)
			
			sense = word + '.' + pos + '.' + x

			try:
				obj = wn.synset(sense)
				hyper = lambda s: s.hypernyms()
				list_hyper = list(obj.closure(hyper))
				for elmt in list_hyper:
					if elmt.name() not in hypernyms:
						hypernyms.append(elmt.name())
			except Exception as e:
				# print(word, e)
				continue

		return hypernyms

# print(vn.classids(lemma='take'))
# print(vn.classids(lemma='speak'))
# print(vn.classids(lemma='say'))