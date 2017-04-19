import json
from pprint import pprint

def extract_character(filename, character_name):
	with open(filename) as data_file:
		data = json.load(data_file)

	for i in range(len(data['characters'])):
		names = data['characters'][i]['names']

		for name in names:
			if character_name in name['n']:
				return data['characters'][i]

	return None

pprint(extract_character('austen.book.id', 'Elinor'))