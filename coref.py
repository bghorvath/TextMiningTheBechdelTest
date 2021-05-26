# %%
## Importing packages

import numpy as np
import pandas as pd
import re
import json
import os
import sys
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize

# %%
## Reading filtered movies_df from pickle

open_file = open('pickles/fmovies_df.pkl', "rb")
fmovies_df = pickle.load(open_file)

# %%
## Load example movie_json

with open('data/movie_dialogues.txt', 'r') as f:
    for i, line in enumerate(f):
        if i == 9:
            movie_json = json.loads(line)

# %%

# exsent = 'My sister, Angela has a dog. She loves him. She would do everything for David as well. She loves him too.'

# doc = nlp(exsent)

# %%

# char_set = []
# a =  []
# combine = {}

# for pg in movie_json['paragraphs']:
#     for line in pg['dialogues']:
#         if line['character'] == 'NA':
#             doc = nlp(line['line'])
#             for i in doc._.coref_clusters:
#                 key = str(i[0])
#                 if key in char_set:
#                     a.append([key, i.mentions])
            
#             for i in a:
#                 values = [str(j) for j in i[1]]
#                 if i[0] not in combine:
#                     combine[i[0]]=set(values)
#                 else:
#                     combine[i[0]].update(values)

# %%

# span = doc[-2:-1]
# print(span)
# print(span._.is_coref)
# print(span._.coref_cluster.main)
# print(span._.coref_cluster.main._.coref_cluster)
# print(doc._.coref_clusters)
# print(doc._.coref_clusters[1].mentions)
# print(doc._.coref_clusters[1].mentions[-1])
# print(doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)

# %%

import spacy
import neuralcoref
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

# %%

def get_coreference_dict(lines, chars):
    
    movie_json = json.loads(lines)
    # char_json = json.loads(chars)

    movie_index = movie_json['movie_id']
    # char_set = set(char_json['char_set'])

    # Init coreference dict for the movie
    coreference_dict = {}
    
    for pg in movie_json['paragraphs']:

        for dialogue in pg['dialogues']:
            if dialogue['character'] == 'NA':
                doc = nlp(dialogue['line'])
                for i in doc._.coref_clusters:
                    key = str(i[0]).upper()
                    # if key in char_set:
                    mentions = set()
                    for ment in i.mentions:
                        mention = str(ment).lower()
                        if mention in {'he','him','his','himself','boy','man','guy','she','her','herself','girl','woman','gal'}:
                            mentions.add(mention)
                    if mentions != set():
                        coreference_dict[key] = mentions
                
                # for i in coreference_list:
                #     values = [str(j) for j in i[1]]
                #     if i[0] not in coreference_dict:
                #         coreference_dict[i[0]]=set(values)
                #     else:
                #         coreference_dict[i[0]].update(values)
        
        for i, v in coreference_dict.items():
            coreference_dict[i] = list(v)
    
    return {"movie_id": movie_index, "coreference_dict": coreference_dict}

# %%

with open('data/coreference_dict.txt', 'r') as f:
    num_lines = sum(1 for line in f)
    print(f"Currently {num_lines} lines are in")

start_time = time.time()
with open('data/movie_dialogues.txt', 'r') as f, open('data/char_sets.txt', 'r') as g, open('data/coreference_dict.txt', 'a') as h:
    for i, (lines, chars) in enumerate(zip(f, g)):
        if i > num_lines:
            coreference_dict = get_coreference_dict(lines, chars)
            h.write(json.dumps(coreference_dict))
            h.write('\n')
            print(f"{i} done")
        if i > num_lines + 20:
            break
            # if i > 9:
            #     break

print(f"Finished in {time.time()-start_time} seconds")

# %%

with open('data/coreference_dict.txt', 'r') as f:
    for i, line in enumerate(f):
        coreference_dict = json.loads(line)
        if i > 2:
            break

# %%

with open('data/movie_dialogues.txt','r') as f:
    for i, row in enumerate(f):
        if i == 9:
            movie_json = json.loads(row)
            # Init coreference dict for the movie
            coreference_dict = {}

            for pg in movie_json['paragraphs']:

                for dialogue in pg['dialogues']:
                    if dialogue['character'] == 'NA':
                        doc = nlp(dialogue['line'])
                        for i in doc._.coref_clusters:
                            mentions = set()
                            for ment in i.mentions:
                                mention = str(ment).lower()
                                if mention in {'he','him','his','himself','boy','man','guy','she','her','herself','girl','woman','gal'}:
                                    mentions.add(mention)
                            key = str(i[0]).upper()
                            # if key.upper() in char_set:
                            if mentions != set():
                                coreference_dict[key] = mentions
        for i, v in coreference_dict.items():
            coreference_dict[i] = list(v)

# %%

type(coreference_dict)

for asd in coreference_dict:
    asd = list(asd)
    print(asd)

# %%

with open('data/movie_dialogues.txt','r') as f:
    for i, movie in enumerate(f):
        if i == 9:
            movie_json = json.loads(movie)

coreference_dict = {}

for dialogue in movie_json['paragraphs'][46]['dialogues']:
    if dialogue['character'] == 'NA':
        doc = nlp(dialogue['line'])
        for i in doc._.coref_clusters:
            mentions = set()
            for ment in i.mentions:
                mention = str(ment).lower()
                if mention in {'he','him','his','himself','boy','man','guy','she','her','herself','girl','woman','gal'}:
                    mentions.add(mention)
            key = str(i[0]).upper()
            # if key.upper() in char_set:
            if mentions != set():
                coreference_dict[key] = mentions
# %%

# %%
## Importing packages

import numpy as np
import pandas as pd
import re
import json
import os
import sys
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
import spacy
import neuralcoref
# from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)


# with open('data/coreference_dict.txt', 'r') as f:
#     num_lines = sum(1 for line in f)
#     print(f"Currently {num_lines} lines are in")

coref_dir = 'data/coreference_dicts/'
filestems = []

for root, _, files in os.walk(coref_dir):
    for f in files:
        file_path = os.path.join(root,f)
        p = Path(file_path)
        file_stem = int(p.stem)
        filestems.append(file_stem)
    num_movies = max(filestems)

start_time = time.time()
with open('data/movie_dialogues.txt', 'r') as f, open('data/char_sets.txt', 'r') as g:
    for i, (lines, chars) in enumerate(zip(f, g)):
        
        movie_start_time = time.time()
        
        if i > num_movies:
            movie_json = json.loads(lines)
            char_json = json.loads(chars)
            
            char_set = set(char_json['char_set'])

            movie_index = movie_json['movie_id']

            for pg in movie_json['paragraphs']:

                # Init coreference dict for the movie
                coreference_dict = {}

                for dialogue in pg['dialogues']:
                    if dialogue['character'] == 'NA':
                        if len(dialogue['line']) < 50000:
                            doc = nlp(dialogue['line'])
                            for i in doc._.coref_clusters:
                                key = str(i[0]).upper()
                                # if key in char_set:
                                mentions = set()
                                for ment in i.mentions:
                                    mention = str(ment).lower()
                                    if mention in {'he','him','his','himself','boy','man','guy','she','her','herself','girl','woman','gal'}:
                                        mentions.add(mention)
                                if mentions != set():
                                    coreference_dict[key] = mentions

                if coreference_dict != {}:
                    for i, v in coreference_dict.items():
                        coreference_dict[i] = list(v)
                    
                    with open(f'data/coreference_dicts/{movie_index}.txt', 'a') as h:
                        h.write(json.dumps(coreference_dict))
                        h.write('\n')
        
            print(f"{movie_index} done in {time.time()-movie_start_time} s")

            # if i > num_lines + 20:
            #     break
    
print(f"Finished in {time.time()-start_time} seconds")

# %%

with open('data/movie_dialogues.txt', 'r') as f:
    with open('data/movie_dialogues.txt', 'r') as f, open('data/char_sets.txt', 'r') as g:
        for i, (lines, chars) in enumerate(zip(f, g)):
            if i > 0:
                break
            movie_json = json.loads(lines)

# %%

movie_json['movie_id']
# %%
