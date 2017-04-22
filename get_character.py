characters = ['Elinor', 'Marianne', 'Edward', 'Willoughby', 'Brandon']
filename = 'austen.sense.pg161.txt'

def get_character(filename, characters):
	file_dict = {}

	for c in characters:
		file_dict[c] = open(c+'.txt', 'w')
	
	f = open(filename, 'r')
	lines = f.readlines()

	paragraph = ''
	num_par = 0
	pars = []

	for line in lines:
		if line == '\n':
			num_par += 1
			pars.append(paragraph+line)
			paragraph = ''
		else:
			paragraph += line

	for p in pars:
		for c in characters:
			if c in p:
				file_dict[c].write(p)

# get_character(filename, characters)
