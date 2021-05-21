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

open_file = open('movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

open_file = open('log_df.pkl', "rb")
log_df = pickle.load(open_file)

# %%

input_dir = '/home/bghorvath/Documents/pyprojects/irtm_project/processed/'

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

fmovies_df = fmovies_df[['imdb_id','imdb_title','imdb_year','len']]

# %%
## Writing filtered movies_df to pickle

# open_file = open('fmovies_df.pkl', 'wb')
# pickle.dump(merged, open_file)
# open_file.close()

# %%
## Reading filtered movies_df from pickle

open_file = open('fmovies_df.pkl', "rb")
fmovies_df = pickle.load(open_file)

# %%

plt.hist(fmovies_df['imdb_year'].sort_values(), bins = 100)

# %%

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fmovies_df['imdb_year'].value_counts().sort_index())

# %%

fmovies_df

# %%

# Option A: instead of sent_tokenize, first concat rows that don't have \n between them, and sent_tokenize only after that
# Option B: if [] -> [\n], and after sent_tokenize, this way the list elements can be concat based on not having \n between them
# Option C: look for 

# %%
## Preprocessing movie script

split_pattern = r"[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n)" # r"(EXT[^\\]+?\n)"

for root, _, files in os.walk(input_dir):
    # for file_name in files:
    
    script_sentence_list = []

    file_name = '518.txt'
    
    file_path = os.path.join(root, file_name)

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
        # sentence_list = sent_tokenize(a)
        # for rows in f:
        #     # sentence_list = sent_tokenize(rows)
        #     print(sentence_list)
        #     if sentence_list == []:
        #         sentence_list = [r'\n']
        #     script_sentence_str = r'\n'.join(sentence_list)
        #     # script_sentence_list = sentence_list + script_sentence_list

script_text_split = filter(None.__ne__, script_text_split)
script_text_split = list(script_text_split)

len(script_text_split)

# %%

script_text_split[246]

# %%

script_text_ind = []

for i in script_text_split:
    if re.search(r"^(INT|EXT).+\n$", i):
        script_text_ind.append((i, 'heading'))
    else:
        script_text_ind.append((i, 'text'))

len(script_text_ind)

# %%

script_text_ind[0]

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

# %%

len(paragraphs)

# %%

paragraphs_df = pd.DataFrame(paragraphs, columns = ['heading', 'text'])

# %%

print(paragraphs_df[paragraphs_df['heading'] == 'NA'])
paragraphs_df[paragraphs_df['text'] == 'NA']


# %%

paragraphs_df

# %%

pg = paragraphs_df.iloc[90,1]

# %%

# re.split(pg, 

# %%

pg = r'FINNEGAN\n (annoyed)\n The safety...the safety... \n \n He flicks the safety on and off.\n \n FINNEGAN\n Got it?\n \n TRILLIAN\n (pissed)\n Hey! I didn\'t have to come back.\n \n FINNEGAN\n Yeah you did...\n \n TRILLIAN\n (defensive)\n Right... You have a boat.\n \n FINNEGAN\n Boat or no boat... You woulda come\n back anyway. You\'re that kind of\n gal.\n \n TRILLIAN\n Oh yeah? What kind is that?\n \n FINNEGAN\n The "come back" kind.\n \n TRILLIAN\n How do you know that?\n \n FINNEGAN\n Takes one to know one.\n \n Finnegan\'s small smile makes Trillian acutely uncomfortable.\n \n V.O.\n HEELLPP!!\n \n CUT TO:\n \n 9'

# %%

pg_list = sent_tokenize(pg)

# %%

pg

# %%

# 1, Word tokenize and deal with it as a list - if i-1, and i+1 -> .append(i,CHAR)
# 2, Regex split full text for \n CHAR \n 
    # This won't Deal with the descriptive sentences coming after dialogues
    # But for that we can apply passive / active NLP stuff