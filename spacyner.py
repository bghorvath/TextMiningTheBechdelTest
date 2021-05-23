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

# line = movies_json[0]['paragraphs'][1]['dialogues'][0]['line']

# %%
## TODO Creating character_set for the whole json
# TODO feed to Spacy NER

ner_json = []

with open('data/movie_dialogues.txt', 'r') as f:
    for i, row in enumerate(f):
        if i > 0:
            break
        ner_persons = set()
        start_time = time.time()
        movies_json = json.loads(row)
        movie_id = movies_json['movie_id']
        paragraphs = movies_json['paragraphs']
        for pg in paragraphs:
            dialogues = pg['dialogues']
            for dg in dialogues:
                line = dg['line']
                line_token = sent_tokenize(line)
                for sent_token in line_token:
                    clean_line = re.sub(r"[^\w\s']","",sent_token)
                    doc = nlp(clean_line)
                    for ent in doc.ents:
                        if ent.label_ == 'PERSON':
                            ner_persons.add(ent.lemma_)
        ner_json.append({"movie_id": movie_id, "ner_persons": ner_persons})
        print(f"{i} movie done in {time.time()-start_time} seconds")

# 0 movie done in 157.1984305381775 seconds

# %%

# TODO import function isn't working, but may not need it anymore because of json

# TODO test spacy on NER - already build up pipeline for who says what to whom?

# ? do I finetune spacy NER?

# * try out polyglot

# ! look for better sentence tokenizer / or just use re.split...

# * for the what are they talking about:
    # * instead of sentences, look for whole conversation
    # * maybe even split to conversations instead of dialogues?
    # * do topic modeling for whole conversation