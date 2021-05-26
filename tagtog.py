# %%
## Importing packages

import numpy as np
import os
import sys
import json
import time
import random

# %%

random.seed(100)

random_dialogues = []

with open('data/movie_dialogues.txt', 'r') as f:
    for i, row in enumerate(f):
        # if i > 0:
        #     break
        movie_json = json.loads(row)
        # movie_index = movie_json['movie_id']

        for pg in movie_json['paragraphs']:
            for dialogue in pg['dialogues']:
                if dialogue['character'] != 'NA':
                    if random.random() < 0.01:
                        random_dialogues.append(dialogue)

random_subset = random.sample(random_dialogues, 50)

# %%
