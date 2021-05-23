# %%
## Importing packages

import numpy as np
import pandas as pd
import re
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

# 1, Word tokenize and deal with it as a list - if i-1, and i+1 -> .append(i,CHAR)
# 2, Regex split full text for \n CHAR \n 
    # This won't Deal with the descriptive sentences coming after dialogues
    # But for that we can apply passive / active NLP stuff
# Maybe delete text (between parentheses)

# First NER -> then split by that
# Because script recognizes lots of stuff as people just because shitty layout

# %%
## NER for recognizing persons

import spacy

nlp = spacy.load("en_core_web_trf")

# %%
## Clean paragraph input text

char_pattern = r"(\n?\s*(?:[A-Z]+|(?:[A-Z]+\.){1,3}))\s*\n"

#pg = "Stacy's mom is coming over the weekend to see her daughter."

pg_clean = re.sub(char_pattern,r"\1:",pg)
pg_clean = re.sub(r"[^a-zA-Z0-9.\-,;:!?()'\"\s]","",pg_clean)
pg_clean = re.sub(r"\s+"," ", pg_clean)

for f in re.findall("([A-Z]{2,})", pg_clean):
    pg_clean = pg_clean.replace(f, f.title())

pg_clean

# %%

doc = nlp(pg_clean)

# %%

ner_persons = set()

# for ent in doc.ents:
#     print(ent.text)
#     if ent.label_ == 'PERSON':
#         ner_persons.add(ent.lemma_)

# ner_persons
a_set = set()
for token in doc:
    if token.tag_ == 'NNP':
        a_set.add(token.text)
print(a_set)


# %%

from spacy import displacy

displacy.render(doc, style="dep")

# %%

# So first want to use NER to recognize names in text
# From experiments:
    # it seems like messy textual data doesn't really allow for proper name recognition
    # So maybe first clean data of \n's, then apply NER, collect names, and apply re.split to only words appearing in set

# %%

## Need to NER tagtog - do we? It's not ML after all - ask Kata

## Next up: assign passive / active sentences
    ## After that
    #   Name normalization
    #   Gender recognition
    #   Who they talk to - next person in line talking
    #   Coreferences inside text?
    #   What they are talking about
        # Maybe topic modelling?