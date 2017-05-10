import os
import re

# PATH = os.getcwd()
# folder = 'victorian_xml/victorian_xml'

# current_path = os.path.join(PATH, folder)

# files = os.listdir(current_path)

# for filename in files:
# 	digit = re.search('\d', filename)
	
# 	if digit:
# 		folder_name = filename[:digit.start()]
# 		directory = os.path.join(current_path, folder_name)

# 		if not os.path.exists(directory):
# 			os.makedirs(directory)

# 		old_filename = os.path.join(current_path, filename)
# 		new_filename = os.path.join(directory + '/' + filename)
# 		print(old_filename, new_filename)
# 		os.rename(old_filename, new_filename)

folder = 'character_json1'
files = os.listdir(folder)
data = 'data1.csv'

names = [f[:-5] for f in files]
# print(names)
ff = open(data, 'r')
lines = ff.readlines()

for line in lines:
	cc = line.split(',')

	if cc[0] not in names:
		print(cc[0], cc[1])