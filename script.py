import os
import subprocess

folder = 'victorian_characters'

sub1 = os.listdir(folder)

for s1 in sub1:
	sub2 = os.listdir(folder + '/' + s1)
	

	for s2 in sub2:
		sub3 = os.listdir(folder + '/' + s1 + '/' + s2)
		print(sub3)

		for afile in sub3:
			afile = folder + '/' + s1 + '/' + s2 + '/' + afile
			command = 'java -cp * -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ' + afile
			subprocess.call(command.split())

		# break
	# break
