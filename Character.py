import xml.etree.ElementTree as ET
import os
import json
from pprint import pprint
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize

class Character:
	def __init__(self, name, possible_names=None):
		self.name = name
		self.persona = {'agent':[], 'patient':[], 'poss':[], 'mod':[], 'a':[], 'n':[], 'v':[]}
		self.possible_names = possible_names
		self.vector = {}

		self.E = 0 # extroversion
		self.A = 0 # agreeableness
		self.C = 0 # conscientiousness
		self.O = 0 # openness
		self.S = 0 # stability

		self.zE = 0 # extroversion
		self.zA = 0 # agreeableness
		self.zC = 0 # conscientiousness
		self.zO = 0 # openness
		self.zS = 0 # stability

		# meta 
		self.gender = 0
		self.valence = 0 # good or evil
		self.salience = 0 # major or minor character

		self.quotes = []
		self.dialogue_tag = []
		self.sense = {'a':[], 'n':[], 'v':[]}
		self.adjectives = []
		self.adverbs = []

	def isEmpty(self):
		c1 = len(self.persona['agent']) == 0
		c2 = len(self.persona['patient']) == 0
		c3 = len(self.persona['poss']) == 0 
		c4 = len(self.persona['mod']) == 0
		c5 = len(self.quotes) == 0

		return c1 and c2 and c3 and c4 and c5
		
adverb_dict = {}

class Book:
	def __init__(self, title):
		self.title = title
		self.author = ''
		self.book_file = ''

		# dictionary with formal names as keys and Character object
		self.character_list = {} 
		# self.persona = {}
		self.char_sent = {}
		self.char_quote = {}

	def process_quotes(self, quote_file):
		f = open(quote_file, 'r')
		sw = stopwords.words('english')

		# print(sw)
		print(quote_file)

		text = f.read()
		reverse_names = self.reverse_possible_names()

		tt = text.split('================================')
		# print(tt)
		print(len(tt))

		for t in tt:
			if len(t) > 0:
				t = t.strip()
				tmp = t.split('--------------------------------')
				
				dialogue_tag = tmp[0].strip()
				sent = sent_tokenize(dialogue_tag)

				if len(sent) > 1:
					dialogue_tag = sent[0]
				# print(dialogue_tag, sent)

				quote = tmp[1].strip().split('\n')

				for name in reverse_names.keys():
					d = dialogue_tag.split(' ')

					# character is the speaker
					if len(d) > 2:
						length_name = len(name.split(' '))
						orig_name = reverse_names[name]
						
						# if 'Mr.' in dialogue_tag or 'Jarvis' in dialogue_tag or 'Lorry' in dialogue_tag:
						# print(name, ':', dialogue_tag)

						if (length_name == 1 and (name in d[0] or name in d[1])) or (length_name > 1 and (name in ' '.join(d[0:length_name]) or name in ' '.join(d[1:length_name+1]))):
							charObj = self.character_list[orig_name]
							
							if quote not in charObj.quotes and dialogue_tag not in charObj.dialogue_tag:
								charObj.quotes.extend(quote)
								charObj.dialogue_tag.append(dialogue_tag)

		
		for c in self.character_list.keys():
			chara = self.character_list[c]
			print(c)
			dt = chara.dialogue_tag
			print(dt)

			for tag in dt:
				t = nltk.word_tokenize(tag)
				pos = nltk.pos_tag(t)
				
				for pt in pos:
					if pt[1] == 'RB' and pt[0] not in sw and pt[0] != 'X':
						
						if pt[0] in adverb_dict.keys():
							adverb_dict[pt[0]] += 1
						else:
							adverb_dict[pt[0]] = 1

						print(pt[0], end=', ')
			print()

	def process_book_file(self):
		data_file = open(self.book_file, 'r')
		data = json.load(data_file)
		data_file.close()

		character_names = self.reverse_possible_names()

		answer = {}
		tmp = 0

		for i in range(len(data['characters'])):
			names = data['characters'][i]['names']

			for name in names:
				if name['n'] in character_names.keys() and tmp < name['c']:
					formal_name = character_names[name['n']]
					c = self.get_character(formal_name)
					
					agent = data['characters'][i]['agent']
					patient = data['characters'][i]['patient']
					mod = data['characters'][i]['mod']
					poss = data['characters'][i]['poss']

					c.persona['agent'].extend([agent[j]['w'] for j in range(len(agent))])
					c.persona['patient'].extend([patient[j]['w'] for j in range(len(patient))])
					c.persona['mod'].extend([mod[j]['w'] for j in range(len(mod))])
					c.persona['poss'].extend([poss[j]['w'] for j in range(len(poss))])


	# get a coref chain from XML document
	def corefChain(self, filename):
		tree = ET.parse(filename)
		root = tree.getroot()
		doc = root.find('document')
		coref = doc.find('coreference')

		dict_name = {}

		if coref is not None:
			for i, t in enumerate(coref):
				mention = t.iter('mention')

				myname = ''

				for m in mention:
					name = m.find('text').text
					sentence_num = int(m.find('sentence').text)
					head = int(m.find('head').text)

					if len(m.attrib) > 0:
						myname = name
						mm = myname.split()

						if ('Mr' not in mm[0] and 'Miss' not in mm[0]) and mm[0][0].isupper():
							myname = mm[0]

					tup = (head, sentence_num)

					# only change if proper noun
					dict_name[(head,sentence_num)] = myname

		return dict_name
	
	# given an XML file and sentence_id, construct the sentence fully
	def construct_sentence(self, filename, sentence_id):
		tree = ET.parse(filename)
		root = tree.getroot()
		doc = root.find('document')
		sentences = doc.find('sentences')
		sent = sentences.findall('sentence')

		final_sentence = ''

		for s in sent:
			s_id = int(s.attrib['id'])
			
			if s_id == sentence_id:
				token = s.iter('token')

				for t in token:
					word = t.find('word').text
					final_sentence += word + ' '

		return final_sentence

	def getSpeaker(self, filename):
		tree = ET.parse(filename)
		root = tree.getroot()
		doc = root.find('document')
		sentences = doc.find('sentences')
		sent = sentences.findall('sentence')

		speaker = {}

		for s in sent:
			s_id = int(s.attrib['id'])
			token = s.iter('token')

			for t in token:
				t_id = int(t.attrib['id'])

				if t.find('Speaker') == None:
					# print('None speaker')
					continue 
				else:
					speak = t.find('Speaker').text
					word = t.find('word').text
					speaker[(word, s_id, t_id)] = speak

		return speaker

	def buildDependencyTrees(self, filename):
		tree = ET.parse(filename)
		root = tree.getroot()
		doc = root.find('document')
		sentences = doc.find('sentences')
		sent = sentences.findall('sentence')

		trees = {}

		for s in sent:
			tree_dict = {}

			s_id = int(s.attrib['id'])
			dependencies = s.findall('dependencies')

			for d in dependencies:
				if d.attrib['type'] == 'enhanced-plus-plus-dependencies':
					dep = d.findall('dep')

					for dp in dep:
						relation = dp.attrib['type']
						governor = dp.find('governor').text
						dependent = dp.find('dependent').text

						# print(governor, dependent)

						dep_id = int(dp.find('dependent').attrib['idx'])
						gov_id = int(dp.find('governor').attrib['idx'])

						if (gov_id, governor) in tree_dict.keys():
							tree_dict[(gov_id, governor)].append((dep_id, dependent, relation))
						else:
							tree_dict[(gov_id, governor)] = [(dep_id, dependent, relation)]

			trees[s_id] = tree_dict

		return trees

	# get pos tag from an XML document
	def getPOSTag(self, filename):
		tree = ET.parse(filename)
		root = tree.getroot()
		doc = root.find('document')
		sentences = doc.find('sentences')
		sent = sentences.findall('sentence')

		postag = {}

		for s in sent:
			s_id = int(s.attrib['id'])
			token = s.iter('token')

			for t in token:
				t_id = int(t.attrib['id'])
				pos = t.find('POS').text
				word = t.find('word').text

				postag[(word, s_id, t_id)] = pos

		return postag

	# reverse possible names
	# returns dictionary:
	# 	Lizzy --> Elizabeth Bennet
	# 	Elizabeth --> Elizabeth Bennet
	# 	Miss Bennet --> Jane Bennet
	# 	Jane --> Jane Bennet
	def reverse_possible_names(self):
		reverse_names = {}

		for character in self.character_list.keys():
			name = self.character_list[character].name

			for possible_name in self.character_list[character].possible_names:
				reverse_names[possible_name] = name

		return reverse_names


	# given a character's name, return the Character object
	def get_character(self, character_name):
		return self.character_list[character_name]

	# get all characters' persona from this book
	def get_persona(self):
		persona = {}
		for key in self.character_list.keys():
			persona[key] = self.character_list[key].persona
		return persona

	def dfs(self, tree, key):
		if key in tree.keys():
			print("root:", key)
			children = tree[key]
			print("\tchildren:", children)

			for child in children:
				self.dfs(tree, (child[0], child[1]))

	# create persona out of xml files
	def create_persona(self, xml_folder):
		xml_files = os.listdir(xml_folder) # get the XML files from the folder
		reverse_names = self.reverse_possible_names() # get possible names
		lmt = WordNetLemmatizer()

		# print(reverse_names)

		for afile in xml_files:
			afile = xml_folder + '/' + afile
			tree = ET.parse(afile)
			root = tree.getroot()

			doc = root.find('document')
			sentences = doc.find('sentences')
			sent = sentences.findall('sentence')

			postag = self.getPOSTag(afile)
			corefs = self.corefChain(afile)
			speaker = self.getSpeaker(afile)
			trees = self.buildDependencyTrees(afile)

			for s in sent:
				s_id = int(s.attrib['id'])
				mytree = trees[s_id]

				for key in mytree.keys():
					(gov_id, governor) = key
					children = mytree[key]

					for child in children:
						(dep_id, dependent, relation) = child

						if relation in ['nsubj', 'nsubjpass', 'dobj', 'nmod:poss']:
							if (dep_id, s_id) in corefs.keys():
								character = corefs[(dep_id, s_id)]
									
								if character in reverse_names.keys():
									sentence = self.construct_sentence(afile, s_id)
									orig_name = reverse_names[character]

									# get the full sentences
									if orig_name in self.char_sent.keys():
										if sentence not in self.char_sent[orig_name] and '``' not in sentence and '"' not in sentence and "''" not in sentence: # not quote
											self.char_sent[orig_name].append(sentence)
									else:
										if '``' not in sentence and '"' not in sentence and "''" not in sentence:
											self.char_sent[orig_name] = [sentence]

									pos = postag.get((governor, s_id, gov_id), '0')
									lemma = ''

									if pos[0] == 'V':
										lemma = lmt.lemmatize(governor, 'v').lower()
									else:
										lemma = lmt.lemmatize(governor).lower()

									# ---------------------------------------------
									# start nsubj
									# ---------------------------------------------
									if relation == 'nsubj':
										# ---------------------------------------------
										# regular agent
										# ---------------------------------------------
										if pos != '0':
											if pos == 'JJ' or ('NN' in pos and pos != 'NNP'):
												self.get_character(reverse_names[character]).persona['mod'].append((lemma, pos))
											else:
												self.get_character(reverse_names[character]).persona['agent'].append((lemma, pos))

										# ---------------------------------------------
										# find the adverb
										# ---------------------------------------------
										if (gov_id, governor) in mytree.keys():
											children2 = mytree[(gov_id, governor)]
											relations = ['advmod', 'xcomp', 'nmod:with', 'nmod:in']

											for child2 in children2:
												rel2 = child2[2]
												dep2 = child2[1]
												dep_id2 = child2[0]

												postagg = postag[(dep2, s_id, dep_id2)]
												
												if rel2 in relations:
													print(orig_name, ':', governor, dep2 + ' (' + postagg + ')', rel2)

													if postagg == 'RB':
														self.get_character(reverse_names[character]).adverbs.append(dep2.lower())
													elif 'VB' in postagg:
														lemma2 = lmt.lemmatize(dep2, 'v').lower()
														self.get_character(reverse_names[character]).persona['agent'].append((lemma2, postagg))
													elif 'NN' in postagg:
														lemma2 = lmt.lemmatize(dep2).lower()
														self.get_character(reverse_names[character]).persona['poss'].append((lemma2, postagg))
													elif 'JJ' in postagg:
														self.get_character(reverse_names[character]).adjectives.append(dep2.lower())

													if 'NN' in postagg:
														children3 = mytree.get((dep_id2, dep2), [])

														for child3 in children3:
															dep_id3 = child3[0]
															dep3 = child3[1]
															rel3 = child3[2]

															postagg3 = postag[(dep3, s_id, dep_id3)]
															
															if rel3 == 'amod':
																print(dep3, dep2)
																self.get_character(reverse_names[character]).adjectives.append(dep3)
										# ---------------------------------------------
										# end find the adverb
										# ---------------------------------------------
						
									# ---------------------------------------------
									# end nsubj
									# ---------------------------------------------
									# start dobj, nsubjpass
									# ---------------------------------------------
									elif relation in ['dobj', 'nsubjpass']:
										# ---------------------------------------------
										# regular patient
										# ---------------------------------------------
										if pos != '0':
											self.get_character(reverse_names[character]).persona['patient'].append((lemma, pos))
									
									# ---------------------------------------------
									# start nmod:poss
									# ---------------------------------------------
									elif relation == 'nmod:poss':
										if pos != '0' and pos != 'NNP':
											self.get_character(reverse_names[character]).persona['poss'].append((lemma, pos))
						
										# ---------------------------------------------
										# find the adjective
										# ---------------------------------------------
										if (gov_id, governor) in mytree.keys():
											children2 = mytree.get((gov_id, governor), [])

											for child2 in children2:
												rel2 = child2[2]
												dep2 = child2[1]
												dep_id2 = child2[0]

												postagg = postag[(dep2, s_id, dep_id2)]
												
												if rel2 == 'amod':
													print(orig_name, ':', dep2, governor + ' (' + postagg + ')', rel2)
													self.get_character(reverse_names[character]).adjectives.append(dep2)


# ===========================================================================================
# Here is the main program.

# Run this file: 
# > python Character.py [gold standard file] [XML folder]

# Example:
# > python Character.py data.csv XML

# The structure of [XML folder] should be like this: 
# - XML/ 
# 	- book_title1/
# 		- book_title1_0001.xml
# 		- book_title1_0002.xml
# 		- ...
# 	- book_title2/
# 		- book_title2_0001.xml
# 		- book_title2_0002.xml
# 		- ...
#  	- ...

# The outputs of this program are JSON files and an error file.
# For example: "A Tale of Two Cities.json", "A Little Princess.json", etc.
# The error file contains the characters that don't have personas. 
# ===========================================================================================
def main():
	args = sys.argv

	if len(args) != 3:
		print("How to run this file: python Character.py [gold standard file] [XML folder]")
		return

	gold_standard = args[1]
	xml_folder = args[2]

	f = open(os.path.join(os.getcwd(), gold_standard), 'r')
	lines = f.readlines()
	f.close()

	error_file = open('error.txt', 'w')

	book_dictionary = {}

	for line in lines[1:]:
		line = line.strip('\n')
		data = line.split(',')
		character_name = data[0]
		book_title = data[1]
		book_id = data[2]
		author = data[3]
		other_names = data[4].split('/')

		gender = data[5]
		valence = data[6]
		salience = data[7]

		_ae = data[9]
		_aa = data[10]
		_ac = data[11]
		_as = data[12]
		_ao = data[13]

		z_ae = data[14]
		z_aa = data[15]
		z_ac = data[16]
		z_as = data[17]
		z_ao = data[18]


		c = Character(character_name, other_names)
		c.E = _ae
		c.A = _aa
		c.C = _ac
		c.S = _as
		c.O = _ao

		c.zE = z_ae
		c.zA = z_aa
		c.zC = z_ac
		c.zS = z_as
		c.zO = z_ao

		c.gender = gender
		c.valence = valence
		c.salience = salience

		# get book title and its associated list of characters
		if book_title in book_dictionary.keys():
			book_dictionary[book_title].character_list[character_name] = c
		else:
			b = Book(book_title)
			b.json = author
			b.book_file = book_id
			b.character_list[character_name] = c
			book_dictionary[book_title] = b
		# print(line)
	

	# going through each book and create personas of the characters
	for key in book_dictionary.keys():
	# for key in ["A Little Princess"]:
		book = book_dictionary[key]
		folder = xml_folder + '/' + book.book_file[:-5]
		
		try:
			book.create_persona(folder)
			book.process_quotes('quotes/'+book.book_file[:-5] + '.txt.quote')

			for character in book.character_list.keys():
				cObj = book.character_list[character]
				if cObj.isEmpty():
					error_file.write(character)
				else:
					persona = cObj.persona

					persona['name'] = cObj.name

					persona['extroversion'] = cObj.E
					persona['agreeableness'] = cObj.A
					persona['conscientiousness'] = cObj.C
					persona['stability'] = cObj.S
					persona['openness'] = cObj.O

					persona['z_extroversion'] = cObj.zE
					persona['z_agreeableness'] = cObj.zA
					persona['z_conscientiousness'] = cObj.zC
					persona['z_stability'] = cObj.zS
					persona['z_openness'] = cObj.zO

					persona['gender'] = cObj.gender
					persona['valence'] = cObj.valence
					persona['salience'] = cObj.salience
					persona['quotes'] = cObj.quotes
					persona['dialogue_tag'] = cObj.dialogue_tag
					persona['sentences'] = book.char_sent[character]
					persona['adjectives'] = cObj.adjectives
					persona['adverbs'] = cObj.adverbs

					# print(persona)

					outfile = open('character_json/' + character + '.json', 'w')
					json.dump(persona, outfile)
					outfile.close()

		except Exception as e:
			error_file.write('Exception ' + str(e) + ' for file ' + book.title + '\n')
			# print(e)
			continue

	# for x in sorted(adverb_dict, key=adverb_dict.get, reverse=True):
	# 	print(x, adverb_dict[x])

	error_file.close()

main()
