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
## Load dataset of male words

with open('data/gendered_words/gendered_words.json') as f:
    male_words = [i['word'] for i in json.load(f) if i['gender'] == 'm']

for i, word in enumerate(male_words):
    male_words[i] = ' '+word+' '

male_words = set(male_words)

# %%

with open('data/movie_gdialogues.txt','r') as f, open('data/coreference_dict.txt','r') as g, \
    open('data/female_convos.txt','w') as h:
    for movie, coref in zip(f,g):
        # Go through each movie
        
        movie_json = json.loads(movie)
        gender_dict = json.loads(coref)

        movie_index = movie_json['movie_id']
        
        ## Init male words for the certain movie
        movie_male_words = male_words.copy()
        
        ## Add movie-specific male words (character names)
        movie_specific_male_words = {' '+str(key).title()+' ' for key, value in gender_dict['char_genders'].items() if value == 1}
        movie_male_words.update(movie_specific_male_words)
        
        # Init list for the movie to append with female convos
        convo_topic = []

        for pg in movie_json['paragraphs']:
            # Go through each paragraph

            # Init confirmed_convos set to keep track of already seen lines to avoid duplicate convos
            confirmed_convos = set()
            
            for i, dialogue in enumerate(pg['dialogues']):
                # if current line is female
                if dialogue['gender'] == 0:
                    # if last line wasn't from same character
                    if dialogue['character'] != pg['dialogues'][i-1]['character']:
                        # and that person was also female
                        if pg['dialogues'][i-1]['gender'] == 0:
                            # Take last 2 and next 2 lines
                            female_convo = pg['dialogues'][i-2:i+3]
                            
                            # Init topic
                            male_topic = False
                            
                            # Go through every line
                            for j in female_convo:
                                # that isn't an action
                                if j['character'] != 'NA':
                                    # Clean line and add whitespaces to beginning and end, so the any() function finds the male words
                                    clean_line = re.sub(r"[^\w\s']", "", j['line'])
                                    conv = ' '+clean_line+' '
                                    
                                    # Check if line has already been seen or not
                                    # If yes - break the loop
                                    if conv in confirmed_convos:
                                        topic = 'duplicate' # placeholder value to delete easily
                                        break
                                    else:
                                    # If not, add it to seen lines
                                        confirmed_convos.add(conv)
                                    
                                    # Init male_topic inside loop
                                    male_topic = False
                                    
                                    # If any of the male words appear in any of the lines
                                    # Markt it as a male topic and break loop
                                    topic = next((phrase for phrase in movie_male_words if phrase in conv), "")
                                    if topic != "":
                                        male_topic = True
                                        break
                            
                            convo_topic.append({"female_convo": female_convo, "male_topic": male_topic, "topic": topic})
        
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

        convos = female_conv['female_convos']
        
        male_count = sum([1 for i in convos if i['male_topic'] == True])
        not_count = sum([1 for i in convos if i['male_topic'] == False])

        bechdel.append([movie_index, male_count, not_count])

bechdel_df = pd.DataFrame(bechdel, columns = ['movie_id','male_count','not_count'])
bechdel_df = bechdel_df.set_index('movie_id', drop=True)

bechdel_df

# %%
## Saving into pickle

with open('pickles/bechdel.pkl','wb') as f:
    pickle.dump(bechdel_df, f)

# %%
