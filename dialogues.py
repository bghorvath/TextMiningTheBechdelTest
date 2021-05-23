# %%
## Importing packages

import numpy as np
import pandas as pd
import re
import os
import sys
import json
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

## Assigning headers

script_text_ind = []

for i in script_text_split:
    if re.search(r"^(INT|EXT).*\n$", i, re.IGNORECASE):
        script_text_ind.append((i, 'heading'))
    else:
        script_text_ind.append((i, 'text'))

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

len(paragraphs)

# %%

## Going through paragraphs, collecting person names

# Pattern to collect characters with a line
char_pattern = r"[^a-zA-Z \n](?:[ ]|\n)*\n[ ]*((?:[A-Z']+(?:[ ][A-Z]+)?)|(?:[A-Z]+[. ]){1,3})\s*\n"

# Compile character set for the movie

char_set = set()

for pg in paragraphs:
    char_set.update(set(re.findall(char_pattern, pg[1])))

# Split paragraph into dialogue boxes

dialogues = []

# Go through paragraphs, split into dialogues
for pg in paragraphs:
    char_split = re.split(char_pattern, pg[1])
    
    char_split_ind = []

    # Assign char label to character denotations and text to dialogue boxes
    for i in char_split:
        if i in char_set:
            char_split_ind.append((i, 'char'))
        else:
            char_split_ind.append((i, 'text'))

    # Clean dialogue boxes of whitespaces and special characters
    clean_char_split_ind = []
    
    for i in char_split_ind:
        if i[1] == 'char':
            clean_char_split_ind.append(i)
        if i[1] == 'text':
            line = i[0]
            clean_line = re.sub(r"[^a-zA-Z0-9.\-,;:!?()'\"\s]", "", line)
            clean_line = re.sub(r"\s+"," ", clean_line)
            clean_char_split_ind.append((clean_line, 'text'))

    # Match characters with dialogue boxes
    old_block = ('init', 'text')
    # I collect dialogues for whole movie, not just the paragraph, so init is before the for loop
    # dialogues = []

    for block in clean_char_split_ind:

        if block[1] == 'text' and old_block[1] == 'char':
            dialogues.append((old_block[0], block[0]))
        elif block[1] == 'text' and old_block[1] == 'text':
            dialogues.append(('NA', block[0]))
        elif block[1] == 'char' and old_block[1] == 'char':
            dialogues.append((old_block[0], 'NA'))

        old_block = block
    
print(len(dialogues))

# %%

movies_json = []
# movies_json['movies'] = []

for movie_i, row in enumerate(fmovies_df.to_numpy()):
    if movie_i > 10:
        break
    
    # Paragraph split pattern
    split_pattern = r"(?:[^a-zA-Z]((?:EXT|EXTERIOR|INT|INTERIOR)[^a-zA-Z][^\\]+?\n))|(?:\n[^a-zA-Z]*?((?:Ext|Exterior|Int|Interior)[^a-zA-Z][^\\]{0,40}?\n))"

    movie_index = int(fmovies_df.iloc[movie_i,:].name)

    file_name = str(movie_index)+'.txt'
    
    script_sentence_list = []

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
    char_pattern = r"[^a-zA-Z \n](?:[ ]|\n)*\n[ ]*((?:[A-Z']+(?:[ ][A-Z]+)?)|(?:[A-Z]+[. ]){1,3})\s*\n"

    # Compile character set for the movie

    char_set = set()

    for pg in paragraphs:
        char_set.update(set(re.findall(char_pattern, pg[1])))

    paragraphs_json = []

    # Go through paragraphs, split into dialogues
    for pg in paragraphs:

        # Split paragraph into dialogue boxes
        dialogues = []

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
                clean_char_split_ind.append((clean_line, 'text'))

        # Match characters with dialogue boxes
        old_block = ('init', 'text')
        # I collect dialogues for whole movie, not just the paragraph, so init is before the for loop
        # dialogues = []

        for block in clean_char_split_ind:

            if block[1] == 'text' and old_block[1] == 'char':
                char = old_block[0]
                text = block[0]
            elif block[1] == 'text' and old_block[1] == 'text':
                char = 'NA'
                text = block[0]
            elif block[1] == 'char' and old_block[1] == 'char':
                char = old_block[0]
                text = 'NA'

            dialogues.append({'character': char, 'line': text})

            old_block = block
        
        paragraphs_json.append({"header": pg[0], "dialogues": dialogues})
    
    movies_json.append(
        {
            "movie_id": movie_index,
            "paragraphs": paragraphs_json
        }
    )

# %%

with open('movie_dialogues.txt', 'w') as outfile:
    json.dump(movies_json, outfile)

# %%

with open('movie_dialogues.txt', 'r') as f:
    movies_json = json.load(f)

# %%

movies_json[10]['paragraphs'][1]['dialogues'][0]['line']

# %%

movies_json[10]['movie_id']
# %%

