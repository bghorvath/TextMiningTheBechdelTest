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
from polyglot.text import Text

# %%
## Reading pickles

input_dir = 'data/processed/'

# %%
## Reading filtered movies_df from pickle

open_file = open('pickles/fmovies_df.pkl', "rb")
fmovies_df = pickle.load(open_file)

# %%

from dialogues import get_dialogues

# %%
## Get example dialogues in clean sentences

movies_json = []

for movie_i, row in enumerate(fmovies_df.to_numpy()):
    
    if movie_i > 10:
        break
    movie_index = int(fmovies_df.iloc[movie_i,:].name)

    movies_json.append(get_dialogues(movie_index))

lines = []

for line in movies_json[9]['paragraphs'][45]['dialogues']:
    lines.append(line['line'])

clean_lines = []

for line in lines: # NOTE Parentheses solved in function, now it's only tripledot
    line_token = sent_tokenize(line) # TODO try another sentence tokenizer
    for sent_token in line_token:
        clean_line = re.sub(r"[^\w\s']","",sent_token)
        clean_lines.append(clean_line)

# %%

with open('data/movie_dialogues.txt', 'r') as f:
    for i, row in enumerate(f):
        if i > 10:
            break
        if i == 0:
            a = row

a = Text(text)
a.pos_tags

# %%

char_list = []

for row in clean_lines:
    text = Text(row, hint_language_code='en')
    char_list.append(text.entities)

# %%

char_list

# %%
