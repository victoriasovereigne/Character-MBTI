import os
import re

def get_character(filename, file_dict, path_to_output):
        for c in file_dict:
               file_dict[c] = path_to_output + '/' + get_first_alias(c) + '.txt'

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
            for c in file_dict:
                aliases = get_aliases(c)
                if any(alias in p for alias in aliases):
                    with open(file_dict[c], 'a') as f:
                        f.write(p)

        seperate_into_paras(path_to_output, file_dict) 
def get_first_alias(c):        
    return get_aliases(c)[0].replace(' ','_') 

def get_aliases(c):
    return c.split(',')

def seperate_into_paras(path, file_dict):
        for c in file_dict:
            num_para = num_paras(file_dict[c])
            num_digits = len(str(num_para))
            cnt = 0

            with open(file_dict[c], 'r') as character_file:
                directory = path + '/' + get_first_alias(c) 
                if not os.path.exists(directory):
                    os.makedirs(directory)
                lines = character_file.readlines()
                paragraph = ''
                for line in lines:
                    if line == '\n':
                        cnt += 1
                        afile = open(directory + '/' + get_first_alias(c) + str(cnt).zfill(num_digits) + '.txt', 'w')
                        afile.write(paragraph)
                        afile.flush()
                        paragraph = ''
                    else:
                        paragraph += line



def num_paras(afile):
    with open(afile) as f:
        num_paras = 0 
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                num_paras += 1
        return num_paras


def character_dictionary(file):
        d = {}
        with open(file) as f:
            for line in f:
                charName = line.rstrip('\n')
                d[charName] = 0 
        return d

def get_path_to_book(book):
    path_to_book = (os.getcwd() + '/victorian_texts/' + book).replace('\\', '/')
    return path_to_book
     
def get_path_to_char_names(folder, book):
    book_basename = os.path.splitext(book)[0]
    path_to_char_names = (os.getcwd() + '/' + folder + '/' + book_basename + '/characters').replace('\\', '/')
    return path_to_char_names

def get_path_to_output(folder,book):
    book_basename = os.path.splitext(book)[0]
    path_to_output = (os.getcwd() + '/' + folder + '/' + book_basename).replace('\\', '/')
    return path_to_output

if __name__ == '__main__':
        books = os.listdir('victorian_texts')
        characters_dir = 'victorian_characters'
        for book in books:
            path_to_book = get_path_to_book(book) 
            path_to_char_names = get_path_to_char_names(characters_dir, book)
            path_to_output = get_path_to_output(characters_dir, book) 

            char_dict = character_dictionary(path_to_char_names)
            get_character(path_to_book, char_dict, path_to_output)
