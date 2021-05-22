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

open_file = open('pickles/imdb_movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

input_dir = 'data/processed/'

# %%
## Looking at character count of the scripts

txtlen = {}

for root, _, files in os.walk(input_dir):
    for f in files:
        
        file_path = os.path.join(root,f)

        p = Path(file_path)
        file_suffix = p.suffix.lower()
        file_stem = p.stem

        if file_suffix == '.txt':
            with open(file_path, 'r', encoding = 'utf8', errors='ignore') as f:
                text = f.read().replace(" ","")
                txtlen[file_stem] = len(text)

txtlen_df = pd.DataFrame.from_dict(txtlen, orient='index', columns = ['len'])

# %%
## Filtering to only scripts that contain at least 10000 characters

txtlen_df_filtered = txtlen_df[txtlen_df.len > 10000]

plt.hist(txtlen_df_filtered['len'].values, bins = 100)

# %%
## Filtering to 90% mean

txtlen_df_90 = txtlen_df_filtered[(txtlen_df_filtered.len < txtlen_df_filtered.len.quantile(0.95)) \
    & (txtlen_df_filtered.len > txtlen_df_filtered.len.quantile(0.05))]

plt.hist(txtlen_df_90['len'].values, bins = 100)

# %%
## Merging movies_df with txtlen_df_90

txtlen_df_90['i'] = txtlen_df_90.index.astype(int)
movies_df['i'] = movies_df.index.astype(int)
fmovies_df = pd.DataFrame.merge(movies_df, txtlen_df_90, how = 'inner', left_on = 'i', right_on = 'i')

fmovies_df = fmovies_df.set_index('i')

fmovies_df = fmovies_df[['imdb_id','imdb_title','imdb_year','imdb_runtime','imdb_genre','len']]

# %%
## Writing filtered movies_df to pickle

# open_file = open('pickles/fmovies_df.pkl', 'wb')
# pickle.dump(fmovies_df, open_file)
# open_file.close()

# %%
## Reading filtered movies_df from pickle

open_file = open('pickles/fmovies_df.pkl', "rb")
fmovies_df = pickle.load(open_file)

# %%

plt.hist(fmovies_df['imdb_year'].sort_values(), bins = 100)

# %%

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fmovies_df['imdb_year'].value_counts().sort_index())

# %%

# Option A: instead of sent_tokenize, first concat rows that don't have \n between them, and sent_tokenize only after that
# Option B: if [] -> [\n], and after sent_tokenize, this way the list elements can be concat based on not having \n between them
# Option C: look for 

# %%
## Counting # of paragraphs per movie

#for i, movie in enumerate(fmovies_df.values):

def get_paragraph_count(i):

    split_pattern = r"[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n)"
    
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
        script_text = re.sub('\x0c|\t','',script_text)
        script_text_split = re.split(split_pattern, script_text)

    script_text_split = filter(None.__ne__, script_text_split)
    script_text_split = list(script_text_split)

    ## Identifying headers and text blocks

    script_text_ind = []

    for i in script_text_split:
        if re.search(r"^(INT|EXT).+\n$", i):
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
    
    return len(paragraphs)

# %%

fmovies_df['paragraph_count'] = fmovies_df.apply(lambda row: get_paragraph_count(row.name), axis = 1)

# %%
## Preprocessing movie script

split_pattern = r"[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n)"

for root, _, files in os.walk(input_dir):

# for movie in fmovies_df.index:
    
#     file_name = str(movie)+'.txt'
    
    script_sentence_list = []

    file_name = '4.txt'
    
    file_path = os.path.join(input_dir, file_name)

    script_text_list = []

    with open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        for row in f:
            if row == '\n':
                script_text_list.append(row)
            elif len(re.findall(r"\w", row)) > 1:
                script_text_list.append(row)
        script_text = ' '.join(script_text_list)
        script_text = re.sub('\x0c|\t','',script_text)
        script_text_split = re.split(split_pattern, script_text)

script_text_split = filter(None.__ne__, script_text_split)
script_text_split = list(script_text_split)

len(script_text_split)

# %%

script_text_ind = []

for i in script_text_split:
    if re.search(r"^(INT|EXT).+\n$", i):
        script_text_ind.append((i, 'heading'))
    else:
        script_text_ind.append((i, 'text'))

script_text_ind[1]

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

pg = paragraphs_df.iloc[130,1]
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
## Make it a bit easy

pg = " \n A Pakistani counter clerk takes one look at the mob enter-\n ing his store and bolts for the rear. A customer exits\n as Nico herds his captives in.\n \n Hands on the counter!\n \n NICO\n \n Three men do it; the fourth is slow."

# %%

import spacy

nlp = spacy.load("en_core_web_sm")

# %%

doc = nlp(pg)

# %%

a_list = []

for token in doc[0:100]:
    a = [token.text, token.lemma_, token.pos_, token.tag_, token.dep_]
    a_list.append(a)
pd.DataFrame(a_list, columns = ['text','lemma','pos','tag','dep'])

# %%

from spacy import displacy

displacy.render(doc, style="dep")

# %%

# So first want to use NER to recognize names in text
# From experiments:
    # it seems like messy textual data doesn't really allow for proper name recognition
    # So maybe first clean data of \n's, then apply NER, collect names, and apply re.split to only words appearing in set

# %%
