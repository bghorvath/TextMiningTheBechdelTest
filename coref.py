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

neuralcoref.add_to_pipe(nlp)

# %%

exsent = 'David took off his shirt. Lana read her book.'

doc = nlp(exsent)

# %%

for ent in doc.ents:
    print([ent, ent._.coref_cluster])
