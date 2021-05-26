# %%
## Importing packages

import numpy as np
import json
import os
import sys
import time
import pickle
from pathlib import Path

from nltk import word_tokenize

# %%

with open('data/gendered_words/gendered_words.json') as f:
    male_words = set([i['word'] for i in json.load(f) if i['gender'] == 'm'])

# %%

with open('data/movie_gdialogues.txt','r') as f, open('data/coreference_dict.txt','r') as g:
    for i, (movie, coref) in enumerate(zip(f,g)):
        if i > 0:
            break
        
        movie_json = json.loads(movie)
        gender_dict = json.loads(row)
        
        ## Init male words for certain movie
        movie_male_words = male_words
        
        ## Add movie-specific male words (character names)
        for key, value in gender_dict['char_genders'].items():
            if value == 1:
                key = key.lower()
                movie_male_words.add(key)
        
        woman_conv = []
        
        for pg in movie_json['paragraphs']:
            for i, dialogue in enumerate(pg['dialogues']):
                if dialogue['gender'] == 0:
                    try:
                        if dialogue['character'] != pg['dialogues'][i+1]['character']:
                            if pg['dialogues'][i+1]['gender'] == 0:
                                woman_conv.append(pg['dialogues'][i-3:i+3])
                    except IndexError:
                        pass
        
        for i in woman_conv:
            print('asd')


# %%

exdial = woman_convo[-1]

# %%

for i in exdial:
    a = word_tokenize(i['line'])

# %%
