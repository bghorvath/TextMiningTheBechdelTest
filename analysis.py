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
## Looking at movies per year

plt.hist(fmovies_df['imdb_year'].sort_values(), bins = 100)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fmovies_df['imdb_year'].value_counts().sort_index())

# %%
## Counting # of paragraphs per movie

def get_paragraph_count(i):

    split_pattern = r"(?:[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n))|(?:\n[^a-zA-Z]*?((?:Ext|Exterior|Int|Interior)[^a-zA-Z][^\\]{0,40}?\n))"
    
    file_name = str(i)+'.txt'

    file_path = os.path.join(input_dir, file_name)

    ## Reading in file and splitting to paragraphs

    script_text_list = []
    with open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        for row in f:
            if row == '\n':
                script_text_list.append(row)
            elif len(re.findall(r"\w", row)) > 1:
                script_text_list.append(row)
        script_text = ' '.join(script_text_list)
        script_text = re.sub('\x0c|\t|\x81|\x80|\x8e|\x85|\x92|\x93|\x94|\x97|\xa0','',script_text)
        script_text_split = re.split(split_pattern, script_text)

    script_text_split = filter(None.__ne__, script_text_split)
    script_text_split = list(script_text_split)

    ## Identifying headers and text blocks

    script_text_ind = []

    for i in script_text_split:
        if re.search(r"^(INT|EXT).+\n$", i, re.IGNORECASE):
            script_text_ind.append((i, 'heading'))
        else:
            script_text_ind.append((i, 'text'))
    
    ## Connecting headers and text blocks

    old_block = ('init', 'text')

    paragraphs = []

    for block in script_text_ind:

        if block[1] == 'text' and old_block[1] == 'heading':
            paragraphs.append((old_block[0], block[0]))
        elif block[1] == 'text' and old_block[1] == 'text':
            paragraphs.append(('NA', block[0]))
        elif block[1] == 'heading' and old_block[1] == 'heading':
            paragraphs.append((old_block[0], 'NA'))

        old_block = block
    
    paragraphs_df = pd.DataFrame(paragraphs, columns = ['heading', 'text'])

    
    return len(paragraphs)

# %%

fmovies_df['paragraph_count'] = fmovies_df.apply(lambda row: get_paragraph_count(row.name), axis = 1)

# %%
## Preprocessing movie script

split_pattern = r"(?:[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n))|(?:\n[^a-zA-Z]*?((?:Ext|Exterior|Int|Interior)[^a-zA-Z][^\\]{0,40}?\n))"

for root, _, files in os.walk(input_dir):

# for movie in fmovies_df.index:
    
#     file_name = str(movie)+'.txt'
    
    script_sentence_list = []

    file_name = '95.txt'
    
    file_path = os.path.join(input_dir, file_name)

    script_text_list = []

    with open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        for row in f:
            if row == '\n':
                script_text_list.append(row)
            elif len(re.findall(r"\w", row)) > 1:
                script_text_list.append(row)
        script_text = ' '.join(script_text_list)
        script_text = re.sub('\x0c|\t|\x81|\x80|\x8e|\x85|\x92|\x93|\x94|\x97|\xa0','',script_text)
        script_text_split = re.split(split_pattern, script_text)

script_text_split = filter(None.__ne__, script_text_split)
script_text_split = list(script_text_split)

len(script_text_split)

# %%

script_text_ind = []

for i in script_text_split:
    if re.search(r"^(INT|EXT).*\n$", i, re.IGNORECASE):
        script_text_ind.append((i, 'heading'))
    else:
        script_text_ind.append((i, 'text'))

# script_text_ind[1]

# %%

old_block = ('init', 'text')

paragraphs = []

for block in script_text_ind:

    if block[1] == 'text' and old_block[1] == 'heading':
        paragraphs.append((old_block[0], block[0]))
    elif block[1] == 'text' and old_block[1] == 'text':
        paragraphs.append(('NA', block[0]))
    elif block[1] == 'heading' and old_block[1] == 'heading':
        paragraphs.append((old_block[0], 'NA'))

    old_block = block

len(paragraphs)

# %%

paragraphs_df = pd.DataFrame(paragraphs, columns = ['heading', 'text'])

print(paragraphs_df[paragraphs_df['heading'] == 'NA'])
print(paragraphs_df[paragraphs_df['text'] == 'NA'])

paragraphs_df

# %%

pg = paragraphs_df.iloc[30,1]
pg

# %%

char_pattern = r"\n?\s*([A-Z]+|(?:[A-Z]+\.){1,3})\s*\n"

char_split = re.split(char_pattern, pg)

char_split

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

pg = "Stacy's mom is coming over the weekend to see her daughter."

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

for ent in doc.ents:
    print(ent.text)
    if ent.label_ == 'PERSON' and ent.text not in ner_persons:
        ner_persons.add(ent.lemma_)

ner_persons
# for token in doc:
#     a = [token.text, token.lemma_, token.pos_, token.tag_, token.dep_]
#     a_list.append(a)


# %%

from spacy import displacy

displacy.render(doc, style="dep")

# %%

# So first want to use NER to recognize names in text
# From experiments:
    # it seems like messy textual data doesn't really allow for proper name recognition
    # So maybe first clean data of \n's, then apply NER, collect names, and apply re.split to only words appearing in set

# %%

