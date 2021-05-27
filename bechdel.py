# %%
## Importing packages

import numpy as np
import pandas as pd
import json
import re
import os
import sys
import time
import pickle
from pathlib import Path

# %%

with open('data/gendered_words/gendered_words.json') as f:
    male_words = [i['word'] for i in json.load(f) if i['gender'] == 'm']

for i, word in enumerate(male_words):
    male_words[i] = ' '+word+' '

male_words = set(male_words)

# %%

with open('data/movie_gdialogues.txt','r') as f, open('data/coreference_dict.txt','r') as g, \
    open('data/female_convos.txt','w') as h:
    for movie, coref in zip(f,g):
        # if i > 10:
        #     break
        
        movie_json = json.loads(movie)
        gender_dict = json.loads(coref)

        movie_index = movie_json['movie_id']
        
        ## Init male words for certain movie
        movie_male_words = male_words.copy()
        
        ## Add movie-specific male words (character names)
        for key, value in gender_dict['char_genders'].items():
            if value == 1:
                key = ' '+str(key).title()+' '
                movie_male_words.add(key)
        
        convo_topic = []

        for pg in movie_json['paragraphs']:

            # Init confirmed conv set, so that the script keeps track of lines that have already been marked as male or not
            # to avoid duplicates
            confirmed_convos = set()
            
            for i, dialogue in enumerate(pg['dialogues']):
                if dialogue['gender'] == 0:
                    if dialogue['character'] != pg['dialogues'][i-1]['character']:
                        if pg['dialogues'][i-1]['gender'] == 0:
                            female_convo = pg['dialogues'][i-2:i+3]
                            
                            topic = "not"
                            
                            for j in female_convo:
                                if j['character'] != 'NA':
                                    clean_line = re.sub(r"[^\w\s']", "", j['line'])
                                    conv = ' '+clean_line+' ' # ! .lower()
                                    
                                    if conv in confirmed_convos:
                                        topic = 'duplicate'
                                        break
                                    else:
                                        confirmed_convos.add(conv)
                                    
                                    topic = "not"
                                    
                                    if any(phrase in conv for phrase in movie_male_words):
                                        topic = "male"
                                        # print(conv)
                                        break
                            
                            convo_topic.append({"female_convo": female_convo, "topic": topic})
        
        convo_topic = [i for i in convo_topic if i['topic'] != 'duplicate']
        
        female_convos = json.dumps({"movie_id": movie_index, "female_convos": convo_topic})
        
        h.write(female_convos)
        h.write('\n')


# %%
## Compile bechdel dataframe

bechdel = []

with open('data/female_convos.txt','r') as f:
    for i, row in enumerate(f):

            female_conv = json.loads(row)
            
            movie_index = female_conv['movie_id']

            convs = female_conv['female_convos']
            
            male_count = sum([1 for i in convs if i['topic'] == 'male'])
            not_count = sum([1 for i in convs if i['topic'] == 'not'])

            bechdel.append([movie_index, male_count, not_count])

bechdel_df = pd.DataFrame(bechdel, columns = ['movie_id','male_count','not_count'])
bechdel_df = bechdel_df.set_index('movie_id', drop=True)

bechdel_df

# %%

with open('data/female_convos.txt','r') as f:
    for i, row in enumerate(f):
        female_conv = json.loads(row)
        
        movie_index = female_conv['movie_id']
        
        if movie_index == 2175:
            for i, conv in enumerate(female_conv['female_convos']):
                print(conv)

# %%
