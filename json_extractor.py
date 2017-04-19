import json
from pprint import pprint

with open('austen.book.id') as data_file:    
    data = json.load(data_file)

# pprint(data)
print(data['characters'][1])