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

file2 = 'data435.csv'

def convert_name(filename=file2):
	ff = open(file2, 'r')
	lines = ff.readlines()

	for line in lines:
		s = line.split(',')
		name = s[0]

		idx = []
		for i, c in enumerate(name):
			if i > 0 and ' ' not in name:
				if c.isupper():
					idx.append(i)
		# print(idx)

		tmp = ''
		prev = 0
		for i in idx:
			tmp += name[prev:i] + ' '
			prev = i

		tmp += name[prev:]

		print(tmp + ',' + ','.join(s[1:]).strip())

convert_name()
