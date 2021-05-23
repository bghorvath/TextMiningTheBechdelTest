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

def get_paragraph_dialog_count(movie_index):

    # Paragraph split pattern
    split_pattern = r"(?:[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n))|(?:\n[^a-zA-Z]*?((?:Ext|Exterior|Int|Interior)[^a-zA-Z][^\\]{0,40}?\n))"

    file_name = str(movie_index)+'.txt'
    
    script_sentence_list = []

    file_path = os.path.join(input_dir, file_name)

    script_text_list = []

    with open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        for row in f:
            if row == '\n':
                script_text_list.append(row)
            elif len(re.findall(r"[a-zA-Z]", row)) > 1:
                script_text_list.append(row)
        script_text = ' '.join(script_text_list)
        script_text = re.sub('\x0c|\t|\x81|\x80|\x8e|\x85|\x92|\x93|\x94|\x97|\xa0','',script_text)
        script_text_split = re.split(split_pattern, script_text)

    script_text_split = filter(None.__ne__, script_text_split)
    script_text_split = list(script_text_split)

    ## Assigning headers

    script_text_ind = []

    for row in script_text_split:
        if re.search(r"^(INT|EXT).*\n$", row, re.IGNORECASE):
            script_text_ind.append((row.replace('\n', ''), 'heading'))
        else:
            script_text_ind.append((row, 'text'))

    ## Matching headings with texts

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

    ## Going through paragraphs, collecting person names

    # Pattern to collect characters with a line
    char_pattern = r"[^a-zA-Z0-9 \n](?:[ ]|\n)*\n[ ]*((?:[A-Z']+(?:[ ][A-Z]+)?)|(?:[A-Z]+[. ]){1,3})\s*\n"

    # Compile character set for the movie

    char_set = set()

    for pg in paragraphs:
        char_set.update(set(re.findall(char_pattern, pg[1])))

    # Init dialogues: will contain all dialogues of the movie
    dialogues = []

    # Go through paragraphs, split into dialogues
    for pg in paragraphs:

        # Split paragraph into dialogue boxes
        

        char_split = re.split(char_pattern, pg[1])
        
        char_split_ind = []

        # Assign char label to character denotations and text to dialogue boxes
        for row in char_split:
            if row in char_set:
                char_split_ind.append((row, 'char'))
            else:
                char_split_ind.append((row, 'text'))

        # Clean dialogue boxes of whitespaces and special characters
        clean_char_split_ind = []
        
        for row in char_split_ind:
            if row[1] == 'char':
                clean_char_split_ind.append(row)
            if row[1] == 'text':
                line = row[0]
                clean_line = re.sub(r"[^a-zA-Z0-9.\-,;:!?()'\"\s]", "", line)
                clean_line = re.sub(r"\s+"," ", clean_line)
                clean_line = re.sub(r"^\s","", clean_line)
                clean_char_split_ind.append((clean_line, 'text'))

        # Match characters with dialogue boxes
        old_block = ('init', 'text')

        for block in clean_char_split_ind:

            if block[1] == 'text' and old_block[1] == 'char':
                char = old_block[0]
                text = block[0]
                dialogues.append({'character': char, 'line': text})
            elif block[1] == 'text' and old_block[1] == 'text':
                char = 'NA'
                text = block[0]
                dialogues.append({'character': char, 'line': text})
            elif block[1] == 'char' and old_block[1] == 'char':
                char = old_block[0]
                text = 'NA'
                dialogues.append({'character': char, 'line': text})

            old_block = block

    return len(paragraphs), len(dialogues)

# %%

fmovies_df[['paragraph_count','dialogue_count']] = fmovies_df.apply(lambda row: get_paragraph_dialog_count(row.name), axis = 1)

# %%

open_file = open('pickles/fmovies_analysis_df.pkl', 'wb')
pickle.dump(fmovies_df, open_file)
open_file.close()

# %%
