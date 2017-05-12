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
		word_freq_file=None, pos_tag_file=None, sense_freq_file=None, word_hypernyms_file=None,
		abstraction_only=False, get_sense=False, sense_filter={}):
		self.json_folder = json_folder
		self.corpora = {'agent':None, 'patient':None, 'mod':None, 'poss':None, 'all':None, 'a': None, 'n': None, 'v': None, 'sense_all':None}
		self.documents = {'agent':[], 'patient':[], 'mod':[], 'poss':[], 'all':[], 'a': [], 'n': [], 'v': [], 'sense_all':[]}

		# self.corpora_all = None
		
		self.sense_filter = sense_filter
		self.get_sense = get_sense
		self.no_above = no_above
		self.no_below = no_below

		self.character_list = {}

		json_files = os.listdir(json_folder)
		lmt = WordNetLemmatizer()

		agent = []
		patient = []
		mod = []
		poss = []

		sense_v = []
		sense_n = []
		sense_a = []

		self.modes = ['agent', 'patient', 'mod', 'poss']

		self.pos_tag = {}
		self.word_freq = {}
		
		self.sense_freq = {'a':{}, 'n':{}, 'v':{}} # total sense frequency in all documents
		self.word_hypernyms = {'a':{}, 'n':{}, 'v':{}}

		self.initialize_pos_freq(json_folder, word_freq_file, pos_tag_file, sense_freq_file, word_hypernyms_file)
		print(len(self.sense_freq['v']))

		for afile in json_files:
			# print("Processing file", afile)
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
								# print("Getting hypernyms...")
								if 'NN' in pos:
									hypernyms = self.word_hypernyms['n'][w]

									if len(hypernyms) > 0:
										for h in hypernyms:
											if self.sense_freq['n'][h] >= sense_filter['n']:
												sense_sense = self.get_hypernyms(h[:-6], 'n', higher=True)		

												if 'person_sense' not in sense_sense:
													c.persona['n'].append(h)

										# maxh = hypernyms[0]
										# maxc = self.sense_freq['n'][maxh]
										# sense_sense = self.get_hypernyms(maxh[:-6], 'n', higher=True)
										# # print(maxh[:-6], sense_sense)

										# for h in hypernyms[1:]:
										# 	tmp = self.sense_freq['n'][h]
										# 	if tmp > maxc:
										# 		maxc = tmp
										# 		maxh = h
										# 		sense_sense = self.get_hypernyms(maxh[:-6], 'n', higher=True)

										# if 'person_sense' not in sense_sense:
										# 	c.persona['n'].append(maxh)

								elif 'JJ' in pos:
									hypernyms = self.word_hypernyms['a'][w]

									if len(hypernyms) > 0:
										for h in hypernyms:
											if self.sense_freq['a'][h] >= sense_filter['a']:
												c.persona['a'].append(h)
										# maxh = hypernyms[0]
										# maxc = self.sense_freq['a'][maxh]

										# for h in hypernyms[1:]:
										# 	tmp = self.sense_freq['a'][h]
										# 	if tmp > maxc:
										# 		maxc = tmp
										# 		maxh = h

										# c.persona['a'].append(maxh)

								elif 'VB' in pos:
									hypernyms = self.word_hypernyms['v'][w]

									if len(hypernyms) > 0:
										for h in hypernyms:
											if self.sense_freq['v'][h] >= sense_filter['v']:
												c.persona['v'].append(h)
										# maxh = hypernyms[0]
										# maxc = self.sense_freq['v'][maxh]

										# for h in hypernyms[1:]:
										# 	tmp = self.sense_freq['v'][h]
										# 	if tmp > maxc:
										# 		maxc = tmp
										# 		maxh = h

										# c.persona['v'].append(maxh)								

							# -----------------------------
							# if we filter out physical entity
							# -----------------------------
							if abstraction_only and 'NN' in pos:
								# print("Filtering out physical entity...")
								hypernyms = self.get_hypernyms(w, 'n', higher=True)

								if len(hypernyms) > 0 and 'physical_entity_sense' not in hypernyms:
									c.persona[m].append(w)
							else:
								c.persona[m].append(w)
						# end if
					# end for
				# end for

				# print(c.name, c.persona)

				agent.append(c.persona['agent'])
				patient.append(c.persona['patient'])
				mod.append(c.persona['mod'])
				poss.append(c.persona['poss'])

				sense_a.append(c.persona['a'])
				sense_n.append(c.persona['n'])
				sense_v.append(c.persona['v'])

				self.character_list[name] = c
				# print(c.name, c.persona)

			except Exception as e:
				print(afile, e)

		self.documents['agent'] = agent
		self.documents['patient'] = patient
		self.documents['mod'] = mod
		self.documents['poss'] = poss
		self.documents['all'] = all
		self.documents['a'] = sense_a
		self.documents['v'] = sense_v
		self.documents['n'] = sense_n
		self.documents['sense_all'] = sense_v+sense_a+sense_n

		aall = agent + patient + mod + poss
		
		if get_sense:
			self.corpora['a'] = Dictionary(sense_a)
			self.corpora['n'] = Dictionary(sense_n)
			self.corpora['v'] = Dictionary(sense_v)
			self.corpora['sense_all'] = Dictionary(sense_v+sense_n+sense_a)

			self.corpora['a'].filter_extremes(no_below=no_below, no_above=no_above)
			self.corpora['n'].filter_extremes(no_below=no_below, no_above=no_above)
			self.corpora['v'].filter_extremes(no_below=no_below, no_above=no_above)
			self.corpora['sense_all'].filter_extremes(no_below=no_below, no_above=no_above)

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

	def create_dictionary(self, features):
		word_list = []

		for f in features:
			word_list.extend(self.documents[f])
		
		d = Dictionary(word_list)
		d.filter_extremes(no_below=self.no_below, no_above=self.no_above)
		
		return d


	# =========================================
	# create pos tag and count word frequencies
	# =========================================
	def initialize_pos_freq(self, json_folder, word_freq_file, pos_tag_file, sense_freq_file, word_hypernyms_file):
		print("Initializing POS tags...")
		print("Initializing word frequencies...")
		print("Initializing sense frequencies...")
		print("Initializing word hypernyms...")
		json_files = os.listdir(json_folder)

		print("Checking POS tag file...")
		if pos_tag_file is not None:
			print("exists")
			with open(pos_tag_file, 'r') as f:
				self.pos_tag = json.load(f)
		
		print("Checking word frequencies file...")
		if word_freq_file is not None:
			print("exists")
			with open(word_freq_file, 'r') as f:
				self.word_freq = json.load(f)
		
		print("Checking sense frequencies file...")
		# print(sense_freq_file)

		if sense_freq_file is not None:
			print("exists")
			with open(sense_freq_file, 'r') as f:
				self.sense_freq = json.load(f)
		
		print("Checking word hypernyms file...")
		if word_hypernyms_file is not None:
			print("exists")
			with open(word_hypernyms_file, 'r') as f:
				self.word_hypernyms = json.load(f)

			# print(self.word_hypernyms['a']['happy'])
		# return 

		if word_freq_file is None or pos_tag_file is None or word_hypernyms_file is None or sense_freq_file is None:
			for afile in json_files:
				# print("Processing", afile)
				name = afile[:-5]
				afile = json_folder + '/' + afile

				try:
					f = open(afile, 'r')
					data = json.load(f)

					for m in self.modes:
						for word in data[m]:
							w = word[1].lower()
							pos = word[2]
							self.pos_tag[w] = pos

							if w in self.word_freq.keys():
								self.word_freq[w] += 1
							else:
								self.word_freq[w] = 1

							hypernyms = []
							
							if 'NN' in pos:
								hypernyms = self.get_hypernyms(w, 'n') 
								
								if word_hypernyms_file is None:
									self.word_hypernyms['n'][w] = hypernyms

								if sense_freq_file is None:
									for h in hypernyms:
										if h in self.sense_freq['n'].keys():
											self.sense_freq['n'][h] += 1
										else:
											self.sense_freq['n'][h] = 1

							elif 'JJ' in pos:
								hypernyms= self.get_hypernyms(w, 'a')

								if word_hypernyms_file is None:
									self.word_hypernyms['a'][w] = hypernyms

								if sense_freq_file is None:
									for h in hypernyms:
										if h in self.sense_freq['a'].keys():
											self.sense_freq['a'][h] += 1
										else:
											self.sense_freq['a'][h] = 1

							elif 'VB' in pos:
								hypernyms = self.get_hypernyms(w, 'v') 
								
								if word_hypernyms_file is None:
									self.word_hypernyms['v'][w] = hypernyms

								if sense_freq_file is None:
									for h in hypernyms:
										if h in self.sense_freq['v'].keys():
											self.sense_freq['v'][h] += 1
										else:
											self.sense_freq['v'][h] = 1

				except Exception as e:
					print(afile, e)

			# top = {'a':[], 'n':[], 'v':[]}
			# for x in ['a', 'n', 'v']:
			# 	tokens = {(v,k) for (k,v) in self.sense_freq[x].items()}
			# 	top[x] = [(word, count) for (count, word) in sorted(tokens, reverse=True)]
				# print(top[x])
			
			if pos_tag_file is None:
				p = open('pos_tag.json','w')
				json.dump(self.pos_tag, p)

			if word_freq_file is None:
				p = open('word_freq.json','w')
				json.dump(self.word_freq, p)

			if sense_freq_file is None:
				p = open('sense_freq.json','w')
				json.dump(self.sense_freq, p)

			if word_hypernyms_file is None:
				p = open('word_hypernyms.json','w')
				json.dump(self.word_hypernyms, p)

		# =========================================

	# =========================================================
	# Converting the characters' personas into vectors
	# =========================================================
	def convert_character_to_vector2(self, character, dictionary, features):	
		word_list = []

		for f in features:
			word_list.extend(character.persona[f])

		return dictionary.doc2bow(word_list)

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

			if self.get_sense:
				myall = []
				for p in ['a', 'n', 'v']:
					sense = char.sense[p]
					myall.extend(sense)

					char.vector[p] = self.corpora[p].doc2bow(sense)

				char.vector['sense_all'] = self.corpora['sense_all'].doc2bow(myall)

	# =========================================================
	# save the dictionary into files
	# =========================================================
	def save(self):
		self.corpora['agent'].save('agent.dict')
		self.corpora['patient'].save('patient.dict')
		self.corpora['poss'].save('poss.dict')
		self.corpora['mod'].save('mod.dict')
		self.corpora['all'].save('all.dict')

	def get_hypernyms(self, word, pos, higher=False):
		synsets = wn.synsets(word)
		hypernyms = []

		if pos == 'a' or not higher:
			for sn in synsets:
				name = sn.name()[:-5] + '_sense'
				if name not in hypernyms:
					hypernyms.append(name)

			return hypernyms
		else:
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
						name = elmt.name()[:-5] + '_sense'

						if name not in hypernyms:
							hypernyms.append(name)
				except Exception as e:
					# print(word, e)
					continue

		return hypernyms

# print(vn.classids(lemma='take'))
# print(vn.classids(lemma='speak'))
# print(vn.classids(lemma='say'))