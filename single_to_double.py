import os
import re

folder = 'originalTexts'

files = os.listdir(folder)
regex = r"'[A-Z][a-z]*"
regex2 = r"[.,!?]'"

for afile in files:
	path = (os.getcwd() + '/' + folder + '/' + afile).replace('\\', '/')
	f = open(path, 'r')
	lines = f.readlines()

	newfile = open(afile, 'w')

	for line in lines:
		words = line.split()

		for i, word in enumerate(words):
			if '"' not in word and (re.search(regex, word) or re.search(regex2, word)):
				rep = word.replace("'", '"', 1)

				if re.search(regex2, rep):
					rep = rep[:-1] + '"'

				words[i] = rep

		line = " ".join(words)
		
		newfile.write(line + '\n')

	newfile.close()