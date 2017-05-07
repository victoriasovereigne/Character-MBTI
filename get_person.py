import copy
import xml.etree.ElementTree as ET
import gensim
import nltk
import os
import string
from nltk.corpus import wordnet as wn
# from gensim.models.word2vec import Word2Vec
from gensim import corpora
import math

PATH = os.getcwd()
FOLDER = 'output'
FOLDER2 = 'output2'

PATH_TO_FOLDER = os.path.join(PATH, FOLDER)
PATH_TO_FOLDER2 = os.path.join(PATH, FOLDER2)
punct = string.punctuation

def extract_nouns():
	all_files = os.listdir(PATH_TO_FOLDER)
	nouns = []

	for afile in all_files:
		fullname = os.path.join(PATH_TO_FOLDER, afile)
		f = open(fullname, 'r')
		lines = f.readlines()

		for line in lines:
			words = nltk.word_tokenize(line)
			tags = nltk.pos_tag(words)

			for tagged in tags:
				tag = tagged[1]
				word = tagged[0]

				if tag in ['NN', 'NNS']:
					lemma = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word)
					lemma = lemma.lower()

					if lemma not in nouns and lemma[0] not in punct:
						nouns.append(lemma)

	return sorted(nouns)

# nouns = extract_nouns()

# for noun in nouns:
# 	print(noun)


# remove punctuation from a list of words
def remove_punctuation(words):
	new_words = []

	for word in words:
		if word[0] not in punct:
			new_words.append(word)

	return new_words

sw = nltk.corpus.stopwords.words("english")
sw.extend(['said', "n't", 'then', 'answered', 'thou', 'thee', 'thy', 'so', 'when', 'yes', 'got', 'little', 'asked', 'till', 'go', 'come', 'take', 'see', 'tell','give'])

tags = ['CC','DT', 'IN', 'MD', 'TO', 'PRP', 'PRP$', 'NNP', 'NNPS', 'CD']

def remove_stopwords(words):
	new_words = []

	for word in words:
		if word not in sw:
			new_words.append(word) 

	return new_words

def remove_words_with_tags(words, tags):
	new_words = []
	tagged_words = nltk.pos_tag(words)

	# print(tagged_words)

	for tw in tagged_words:
		word = tw[0]
		tag = tw[1]

		if tag not in tags:
			new_words.append(word)

	return new_words

def words_to_lowercase(words):
	new_words = []

	for word in words:
		new_words.append(word.lower())

	return new_words

def lemmatizer(words):
	new_words = []
	tagged_words = nltk.pos_tag(words)

	for tw in tagged_words:
		word = tw[0]
		tag = tw[1]
		lemma = ''

		if tag[0] == 'V':
			lemma = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word,'v')
		else:
			lemma = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word)

		new_words.append(lemma)

	return new_words

def preprocess(filename):
	full_path_filename = os.path.join(PATH_TO_FOLDER, filename)
	total = 0

	# read the file
	content = open(full_path_filename, 'r')
	lines = content.readlines()

	stories_words = []

	# each line is a sentence
	for line in lines:
		words = nltk.word_tokenize(line)

		# total = total + len(words)

		words = remove_punctuation(words)
		words = words_to_lowercase(words)
		words = remove_stopwords(words)
		words = remove_words_with_tags(words, tags)
		words = lemmatizer(words)

		stories_words.extend(words)

	return stories_words
	# return total

# returns a dict of stories
# key = story title
# value = story words
def preprocess_all():
	all_files = os.listdir(PATH_TO_FOLDER)

	# all_stories = [] #change into dict
	all_stories = {}

	for afile in all_files:
		astory = preprocess(afile)
		# all_stories.append(astory)
		af = ''.join(remove_punctuation(afile[:-4]))
		all_stories[af.lower()] = astory

	return all_stories

# texts is a dict
def create_fileids(texts):
	# create new fileids
    # newfileids = []
    newfileids = {}
    keys = list(texts.keys())

    for i in range(0, len(texts)):
        # newfileids.append(str(i))
        newfileids[i] = keys[i]

    return newfileids

# aa = preprocess_all()

# # for key in aa.keys():
# # 	print(aa, aa[key])

# print(create_fileids(aa))

def generate_lda_topics(texts, numtopics, textlabels, numwords, mydict):
    # map terms to IDs
    gdict = gensim.corpora.Dictionary(texts)

    # and represent the corpus in sparse matrix format, bag-of-words
    corpus = [gdict.doc2bow(text) for text in texts]

    # now we make an LDA object.
    lda_obj = gensim.models.ldamodel.LdaModel(corpus, id2word=gdict, num_topics=numtopics, passes = 20)
    lda_corpus = lda_obj[corpus]
    sim_obj = gensim.similarities.MatrixSimilarity(lda_corpus)

    for docindex, doc in enumerate(lda_corpus):
    	# print('docindex: ', docindex, 'doc: ', doc, 'title: ', mydict[docindex])
    	sims = sim_obj[doc]
    	sims_and_labels = sorted(zip(sims, textlabels), reverse=True)
    	# print( "Similarities for", mydict[textlabels[docindex]])
    	print(mydict[textlabels[docindex]], end=',')
    	
    	for sim, textlabel in sims_and_labels[:3]:
    		if textlabel != textlabels[docindex] and sim > 0:
    			print(mydict[textlabel], sim, end=',')
    	print()

    for index, t in enumerate(lda_obj.print_topics(numtopics, numwords)):
        print( "topic", index, end=': ')
        ts = t[1].split('+')
        for tt in ts:
            tx = tt.strip().split('*')
            print(tx[1]+ '(' + tx[0] + ')', end=", ")
        print('\n')

originals = preprocess_all()  # is a dict
fileids = create_fileids(originals) # is also a dict

generate_lda_topics(list(originals.values()), 10, list(fileids.keys()), 20, fileids)

def collect_words(originals):
	word_dict = {}

	for stories in originals:
		for word in stories:
			if word not in word_dict.keys():
				word_dict[word] = 1
			else:
				word_dict[word] = word_dict[word] + 1

	return word_dict

# word_dict = collect_words(originals)

# for key in sorted(word_dict.items(), key=lambda x:x[1], reverse=True):
# 	print(key)

def identify_person(filename='nouns.txt'):
	f = open(filename, 'r')
	lines = f.readlines()

	persons = []

	for line in lines:
		word = line.rstrip()
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
				# print(n, list_hyper)

				for elmt in list_hyper:
					if 'person.n.01' == elmt.name() and word not in persons:
						persons.append(word)
						# print(n)
						break
			except:
				continue

	return persons

# x = identify_person()

# ====================================
# Identify actors from a folder of stores
# ====================================
def get_actors(folder=PATH_TO_FOLDER):
	files = os.listdir(folder)

	person_file = os.path.join(PATH, 'persons2.txt')
	pp = open(person_file, 'r')

	persons = pp.readlines()

	stories_actors = {}

	for afile in files:
		stories_actors[afile] = []

		myfile = os.path.join(folder, afile)
		f = open(myfile, 'r')
		lines = f.readlines()

		for line in lines:
			words = nltk.word_tokenize(line)

			for person in persons:
				person = person.strip()
				person1 = person.split('.')[0]

				if person1 in words and person not in stories_actors[afile]:
					stories_actors[afile].append(person)

		stories_actors[afile] = sorted(stories_actors[afile])
		print(afile, stories_actors[afile])

	return stories_actors

def cosine(vec1, vec2):
	num = 0
	denum = 1
	a = 0
	b = 0

	for i in range(0, len(vec1)):
		for j in range(0, len(vec2)):
			if vec1[i][0] == vec2[j][0]:
				num = num + vec1[i][1] * vec2[j][1]

	for i in range(0, len(vec1)):
		a = a + vec1[i][1] * vec1[i][1]

	a = math.sqrt(a)

	for i in range(0, len(vec2)):
		b = b + vec2[i][1] * vec2[i][1]

	b = math.sqrt(b)

	denum = a * b

	if denum == 0:
		return -100

	return num/denum

def main():
	act_dict = get_actors()

	out = open('comp1.txt', 'w')
	keys = sorted(act_dict.keys())

	for i in range(0, len(keys)):
		key1 = keys[i]
		for j in range(i+1, len(keys)):
			key2 = keys[j]

			if key1 != key2:
				sim = get_similarity(act_dict[key1], act_dict[key2])
				out.write(key1 + ' ' + key2 + ' ' + str(sim) + '\n')
				print(key1, key2, sim)

	# actors = []
	# for key in sorted(act_dict.keys()):
	# 	print(key, act_dict[key])
	# 	actors.append(act_dict[key])

	# dictionary = corpora.Dictionary(actors)
	# dictionary.save('dictionary.txt')

	# dictionary = corpora.Dictionary.load('dictionary.txt')
	# print(dictionary)
	# print(dictionary.token2id)

def compare(dict_name, dictionary):
	for key in sorted(dict_name.keys()):
		vec1 = dictionary.doc2bow(dict_name[key])

		for key2 in sorted(dict_name.keys()):
			if key != key2:
				vec2 = dictionary.doc2bow(dict_name[key2])
				print(key, key2, cosine(vec1, vec2))

def to_frame(folder=os.path.join(PATH, 'xml')):
	files = os.listdir(folder)

	story_frames = {}
	
	for afile in files:
		try:
			myfile = os.path.join(folder, afile)
			tree = ET.parse(myfile)
			root = tree.getroot()

			text = root.iter('text')
			story_frames[afile] = []

			for child in root.iter('annotationSet'):
				frame = child.get('frameName')
				story_frames[afile].append(frame)
		except:
			continue

	return story_frames

def main2():
	k = to_frame()

	# frames = []

	# for key in k.keys():
	# 	frames.append(k[key])

	# dictionary2 = corpora.Dictionary(frames)
	# dictionary2.save('frametionary.txt')

	dictionary2 = corpora.Dictionary.load('frametionary.txt')

	compare(k, dictionary2)

def get_scores(filename='frames1.txt'):
	f = open(filename, 'r')
	lines = f.readlines()

	for line in lines:
		w = line.split()
		if w[1] != 'Footnotes' and float(w[2]) > 0:
			print(w[2], ',')

# get_scores('comparison.txt')

def wup_hacked(a1, a2):
	if a1 == a2:
		return 1.0
	else:
		return wn.wup_similarity(a1, a2)

# from nltk.corpus import wordnet_ic
# brown_ic = wordnet_ic.ic('ic-brown.dat')
# semcor_ic = wordnet_ic.ic('ic-semcor.dat')

def get_similarity(aa1, aa2):
	same = []
	actors1 = []
	actors2 = []

	for aa in aa1:
		actors1.append(wn.synset(aa))

	for aa in aa2:
		actors2.append(wn.synset(aa))

	# remove the same elements in both lists
	for a1 in actors1:
		for a2 in actors2:
			if a1 == a2:
				same.append(a1)

	for el in same:
		try:
			actors1.remove(el)
			actors2.remove(el)
		except:
			continue

	# print(actors1, actors2)

	shorter = actors1
	longer = actors2

	if len(actors1) > len(actors2): 
		shorter = actors2
		longer = actors1

	length = len(longer) + len(same)

	match = [-1] * len(shorter)
	highest = [0.0] * len(shorter)
	for i in range(0, len(shorter)):
		s = shorter[i]

		for j in range(0, len(longer)):
			l = longer[j]
			score = wup_hacked(s, l)

			if score > highest[i] and j not in match:
				highest[i] = score
				match[i] = j

	# print(match)
	# print(highest)

	answer = (len(same) + sum(highest)) / length
	# print(answer)
	return answer

	# scores = []
	# for i in range(0, len(shorter)):
	# 	scores.append([])
	# 	for j in range(0, len(longer)):
	# 		scores[i].append(wup_hacked(shorter[i], longer[j]))

	# for score in scores:
	# 	print(score)

	# for i in range(0, len(scores)):
	# 	for j in range(0, len(scores[i])):



# actors1 = ['queen.n.02', 'prince.n.01', 'king.n.01', 'princess.n.01']
# actors2 = ['king.n.01', 'princess.n.01', 'nobleman.n.01', 'witch.n.01', 'merchant.n.01']

# get_similarity(actors1, actors2)

# main()

def compare_both(file1='comp1.txt', file2='frames1.txt'):
	# read file 1 and 2
	f1 = open(file1, 'r')
	f2 = open(file2, 'r')

	l1 = f1.readlines()
	l2 = f2.readlines()

	c1 = []
	c2 = []

	compdict = {}

	for line in l1:
		arr = line.split()
		s1 = ''.join(remove_punctuation(arr[0][:-4])).lower()
		s2 = ''.join(remove_punctuation(arr[1][:-4])).lower()
		score = arr[2]
		# c1.append([s1, s2, score])
		compdict[(s1, s2)] = [score]

	for line in l2:
		arr = line.split()
		s1 = ''.join(remove_punctuation(arr[0][:-4])).lower()
		s2 = ''.join(remove_punctuation(arr[1][:-4])).lower()
		score = arr[2]

		if (s1, s2) in compdict.keys() or (s2, s1) in compdict.keys():
			try:
				compdict[(s1, s2)].append(score)
			except:
				try:
					compdict[(s1, s2)].append(score)
				except:
					continue

	for key in sorted(compdict.keys()):
		if len(compdict[key]) > 1:
			print(key[0], key[1], compdict[key], '-->', str((float(compdict[key][0]) + float(compdict[key][1])) / 2))

# compare_both()

def get_most_similar(filename='scores.txt'):
	f = open(filename, 'r')
	lines = f.readlines()

	first = {}
	second = {}

	for line in lines:
		arr = line.split()
		story1 = arr[0]
		story2 = arr[1]
		actor_score = float(arr[2])
		frame_score =  float(arr[3])
		both_score = float(arr[4])

		# check the highest score



# get_most_similar()
