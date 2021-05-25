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
## Reading pickles

input_dir = 'data/processed/'

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

import spacy
import neuralcoref

nlp = spacy.load("en_core_web_sm")

# %%

nlp.remove_pipe('neuralcoref')

neuralcoref.add_to_pipe(nlp) #, conv_dict = [{"Mrs. Brennan": ["mother"]}, {'Jake': ['boy']}])

# %%

exsent = 'My sister, Angela has a dog. She loves him. She would do everything for David as well. She loves him too.'

doc = nlp(exsent)

# %%

lines = []

for line in movie_json['paragraphs'][46]['dialogues']:
    if line['character'] == 'NA':
        lines.append(line['line'])

lines = lines[0:9]
lines = ". ".join(lines)

# %%

doc = nlp(lines)

# %%

doc._.coref_clusters[0]

# %%

for ent in doc.ents:
    print(ent)

# %%
span = doc[-2:-1]
print(span)
print(span._.is_coref)
print(span._.coref_cluster.main)
print(span._.coref_cluster.main._.coref_cluster)

# %%

print(doc._.coref_clusters)
print(doc._.coref_clusters[1].mentions)
print(doc._.coref_clusters[1].mentions[-1])
print(doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)

# %%

with open('data/movie_dialogues.txt', 'r') as f:
    for i, line in enumerate(f):
        if i == 9:
            movie_json = json.loads(line)
    
    for pg in movie_json['paragraphs']:
        for dial in pg['dialogues']:
            print('asd')

# %%

a = [{'character': 'Peter', 'line': 'A'}, {'character': 'Peter', 'line': 'B'}, {'character': 'NA', 'line': 'C'}, {'character': 'NA', 'line': 'D'}, {'character': 'NA', 'line': 'E'}, {'character': 'Park', 'line': 'F'}, {'character': 'Park', 'line': 'G'}]

sents = []

chars = []
lines = []

for i in a:
    chars.append(i['character'])
    lines.append(i['line'])

range_flag = 0
last_char = ''

lenchars = len(chars)

for index, char in enumerate(chars):

    if char != last_char:
        # set new index, else keep previous
        range_flag = index

    if index < lenchars-1:
        # if it's not the last index, look who the next character is
        next_char = chars[index+1]
        # if it's different, close off group
        if char != next_char:
            sents.append({'character': char, 'line': " ".join(lines[range_flag:index+1])})
    else:
        # if there's no dialogue, close off group no matter what
        sents.append({'character': char, 'line': " ".join(lines[range_flag:])})
    
    # iterate to new last_char
    last_char = char


# %%
