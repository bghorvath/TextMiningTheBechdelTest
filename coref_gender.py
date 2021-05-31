# %%
## Importing packages

import numpy as np
import json
import os
import sys
import time
from pathlib import Path

from collections import Counter

# %%
## Importing SpaCy and neuralcoref, creating pipeline

import spacy
import neuralcoref

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

# %%
## Get dictionary of gendered words

!git clone https://github.com/ecmonsen/gendered_words.git data

# %%
## Create gender coreference set

with open('data/gendered_words/gendered_words.json') as f:
    gender_coreference = {i['word'] for i in json.load(f) if i['gender'] in {'f','m'}}

# %%
## Init coreference dictionaries folder

coref_dir = 'data/coreference_dicts/'

# %%
## Collect movies that are already done, so loop won't go through them again

movie_indexes = set()

for root, _, files in os.walk(coref_dir):
    for f in files:
        file_path = os.path.join(root,f)
        p = Path(file_path)
        file_stem = int(p.stem)
        movie_indexes.add(file_stem)

# %%
## Create character-gender dictionaries for each movie

start_time = time.time()
with open('data/movie_dialogues.txt', 'r') as f, open('data/char_sets.txt', 'r') as g:
    for i, (lines, chars) in enumerate(zip(f, g)):
        
        movie_start_time = time.time()

        movie_json = json.loads(lines)

        movie_index = movie_json['movie_id']
        
        # Only for movies that aren't done yet
        if movie_index not in movie_indexes:
        
            char_json = json.loads(chars)
            
            char_set = set(char_json['char_set'])

            for pg in movie_json['paragraphs']:

                # Init coreference dict for the movie
                coreference_dict = {}
                
                # Go through each paragraph
                for dialogue in pg['dialogues']:
                    if dialogue['character'] == 'NA':
                        if len(dialogue['line']) < 50000:
                            doc = nlp(dialogue['line'])
                            for i in doc._.coref_clusters:
                                key = str(i[0]).upper()
                                if key in char_set:
                                    mentions = set()
                                    for ment in i.mentions:
                                        mention = str(ment).lower()
                                        if mention in gender_coreference:
                                            mentions.add(mention)
                                    if mentions != set():
                                        coreference_dict[key] = mentions

                if coreference_dict != {}:
                    for i, v in coreference_dict.items():
                        coreference_dict[i] = list(v)
                    
                    with open(os.path.join(coref_dir, str(movie_index)+'.txt'), 'a') as h:
                        h.write(json.dumps(coreference_dict))
                        h.write('\n')
        
            print(f"{movie_index} done in {time.time()-movie_start_time} s")
    
print(f"Finished in {time.time()-start_time} seconds")

# %%
## Create set for male words

with open('data/gendered_words/gendered_words.json', 'r') as f:
    male = set([i['word'] for i in json.load(f) if i['gender'] == 'm'])

# %%
## Assign gender to characters, create gender coreference dictionary

with open('data/char_sets.txt','r') as f, open('data/coreference_dict.txt','w') as g:
    for row in f:
        char_json = json.loads(row)
        movie_index = char_json['movie_id']

        file_path = os.path.join(coref_dir, str(movie_index)+'.txt')

        char_genders = {}

        try:
            with open(file_path, 'r') as h:
                for i, line in enumerate(h):
                    if line != {}:
                        coref_dict = json.loads(line)
                        for key, value in coref_dict.items():
                            try:
                                char_genders[key] = char_genders[key] + [1 if i in male else 0 for i in value]
                            except KeyError:
                                char_genders[key] = [1 if i in male else 0 for i in value]
                for key, value in char_genders.items():
                    char_genders[key] = Counter(value).most_common(1)[0][0]
        except FileNotFoundError:
            pass
        g.write(json.dumps({"movie_id": movie_index, "char_genders": char_genders}))
        g.write('\n')

# %%
## Create a gendered version of the movie_dialogues.txt

with open('data/movie_dialogues.txt', 'r') as f, open('data/coreference_dict.txt','r') as g, \
    open('data/movie_gdialogues.txt','w') as h:
    for i, (movie, coref) in enumerate(zip(f,g)):

        movie_json = json.loads(movie)
        coref_json = json.loads(coref)

        movie_index = movie_json['movie_id']

        paragraphs = []

        for pg in movie_json['paragraphs']:
            
            dialogues = []

            for dialogue in pg['dialogues']:
                char = dialogue['character']
                try:
                    gender = coref_json['char_genders'][char]
                except KeyError:
                    gender = 'NA'
                dialogue_dict = {'character': char, 'gender': gender, 'line': dialogue['line']}
                dialogues.append(dialogue_dict)
            paragraphs.append({'header': pg['header'], 'dialogues': dialogues})
        
        movie = json.dumps({'movie_id': movie_index, 'paragraphs': paragraphs})
        h.write(movie)
        h.write('\n')

# %%
