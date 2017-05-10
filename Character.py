import xml.etree.ElementTree as ET
import os
import json
from pprint import pprint
import sys
from nltk.stem.wordnet import WordNetLemmatizer

class Character:
	def __init__(self, name, possible_names=None):
		self.name = name
		self.persona = {'agent':[], 'patient':[], 'poss':[], 'mod':[]}
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

	def isEmpty(self):
		return len(self.persona['agent']) == 0 and len(self.persona['patient']) == 0 and len(self.persona['poss']) == 0 and len(self.persona['mod']) == 0

		
class Book:
	def __init__(self, title):
		self.title = title
		self.json = ''
		self.book_file = ''

		# dictionary with formal names as keys and Character object
		self.character_list = {} 
		# self.persona = {}

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

					# print(formal_name)
					
					# print(c.persona['agent'])
					# print(c.persona['patient'])
					# print(c.persona['mod'])
					# print(c.persona['poss'])


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

		# for key in dict_name.keys():
		# 	print(key, dict_name[key])

		return dict_name

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
			# print(corefs)

			for s in sent:
				s_id = int(s.attrib['id'])
				enhanced_dep = s.findall('dependencies')

				for d in enhanced_dep:
					if d.attrib['type'] == 'enhanced-plus-plus-dependencies':
						dep = d.findall('dep')

						potential_pred = ''
						tmp_id = 0

						for dp in dep:
							if dp.attrib['type'] == 'nsubj':
								governor = dp.find('governor').text
								dependent = dp.find('dependent').text
								dep_id = int(dp.find('dependent').attrib['idx'])
								gov_id = int(dp.find('governor').attrib['idx'])

								if (dep_id, s_id) in corefs.keys():
									character = corefs[(dep_id, s_id)]
									# print(character)
									if character in reverse_names.keys():
										
										# print(governor, s_id, gov_id)
										pos = postag.get((governor, s_id, gov_id), '0')
										lemma = ''

										if pos[0] == 'V':
											lemma = lmt.lemmatize(governor, 'v')
										else:
											lemma = lmt.lemmatize(governor)

										if pos != '0':
											if pos == 'JJ' or ('NN' in pos and pos != 'NNP'):
												self.get_character(reverse_names[character]).persona['mod'].append((governor, lemma, pos))
											else:
												self.get_character(reverse_names[character]).persona['agent'].append((governor, lemma, pos))

							elif dp.attrib['type'] == 'dobj':
								governor = dp.find('governor').text
								dependent = dp.find('dependent').text
								dep_id = int(dp.find('dependent').attrib['idx'])
								gov_id = int(dp.find('governor').attrib['idx'])

								if (dep_id, s_id) in corefs.keys():
									character = corefs[(dep_id, s_id)]
									# print(character)
									if character in reverse_names.keys():
										
										# print(governor, s_id, gov_id)
										pos = postag.get((governor, s_id, gov_id), '0')
										lemma = lmt.lemmatize(governor, 'v')
										if pos != '0':
											self.get_character(reverse_names[character]).persona['patient'].append((governor, lemma, pos))

							elif dp.attrib['type'] == 'nmod:poss':
								governor = dp.find('governor').text
								dependent = dp.find('dependent').text
								dep_id = int(dp.find('dependent').attrib['idx'])
								gov_id = int(dp.find('governor').attrib['idx'])

								if (dep_id, s_id) in corefs.keys():
									character = corefs[(dep_id, s_id)]
									# print(character)
									if character in reverse_names.keys():
										
										# print(governor, s_id, gov_id)
										pos = postag.get((governor, s_id, gov_id), '0')
										lemma = lmt.lemmatize(governor)
										if pos == 'NNP':
											print(governor)
										if pos != '0' and pos != 'NNP':
											self.get_character(reverse_names[character]).persona['poss'].append((governor, lemma, pos))



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
	# print(args)

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
		book_json = data[3]
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
			b.json = book_json
			b.book_file = book_id
			b.character_list[character_name] = c
			book_dictionary[book_title] = b
	
	key = "Jane Eyre"
	# book = book_dictionary[key]
	# folder = xml_folder + '/' + book.book_file[:-5]

	# book.create_persona(folder)

	# files = os.listdir(folder)

	# for character in book.character_list.keys():
	# 	cc = book.character_list[character]

	# 	for afile in files:
	# 		f = open(folder + '/' + afile, 'r')
	# 		lines = f.read()

	# 		for name in cc.possible_names:
	# 			if name in lines:
	# 				print(cc.name, afile)

	# going through each book and create personas of the characters
	for key in book_dictionary.keys():
	# for key in ["Jane Eyre"]:
		book = book_dictionary[key]
		folder = xml_folder + '/' + book.book_file[:-5]
		try:
			book.create_persona(folder)

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

					outfile = open('character_json/' + character + '.json', 'w')
					json.dump(persona, outfile)
					outfile.close()

		except Exception as e:
			error_file.write('Exception ' + str(e) + ' for file ' + book.title + '\n')
			continue

	error_file.close()

main()
