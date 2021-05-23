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

from dialogues import get_dialogues

# %%

movies_json = []

for movie_i, row in enumerate(fmovies_df.to_numpy()):
    
    if movie_i > 10:
        break
    movie_index = int(fmovies_df.iloc[movie_i,:].name)

    movies_json.append(get_dialogues(movie_index))

# %%

lines = []

for line in movies_json[9]['paragraphs'][45]['dialogues']:
    lines.append(line['line'])

# %%

clean_lines = []

for line in lines: # NOTE Parentheses solved in function, now it's only tripledot
    line_token = sent_tokenize(line)
    for sent_token in line_token:
        clean_line = re.sub(r"[^\w\s']","",sent_token)
        clean_lines.append(clean_line)

# %%
## NER for recognizing persons

import spacy

nlp = spacy.load("en_core_web_trf")

# %%

doc = nlp(clean_line)

# %%

for ent in doc.ents:
    print(ent.text)
    if ent.label_ == 'PERSON':
        ner_persons.add(ent.lemma_)

# for token in doc:
#     if token.tag_ == 'NNP':
#         a_set.add(token.text)
# print(a_set)


# %%

from spacy import displacy

displacy.render(doc, style="dep")

# %%

for label in nlp.get_pipe("tagger").labels:
    print(label, " -- ", spacy.explain(label))

# %%

text = "Act like an adult... Frigo suddenly jumps out an open window and scampers off into the night."

sent_tokenize(text)
# %%
## TODO Creating character_set for the whole json
# TODO feed to Spacy NER

with open('data/movie_dialogues.txt', 'r') as f:
    for row in f:
        movies_json = json.load(row)

