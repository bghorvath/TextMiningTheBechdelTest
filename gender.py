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

coref_dir = 'data/coreference_dicts/'

male = {'he','him','his','himself','boy','man','guy'}
female = {'she','her','herself','girl','woman','gal'}

# %%

counter = 0

with open('data/coreference_dict.txt','w') as f:
    for root, _, files in os.walk(coref_dir):
        for file in files:
            counter = counter + 1
            file_path = os.path.join(root, file)
            p = Path(file_path)
            
            movie_index = int(p.stem)

            # if counter > 30:
            #     break

            char_genders = {}

            with open(file_path, 'r') as g:
                for i, line in enumerate(g):
                    # if i > 100:
                    #     break
                    if line != {}:
                        coref_dict = json.loads(line)
                        for key, value in coref_dict.items():
                            try:
                                char_genders[key] = char_genders[key] + [1 if i in male else 0 for i in value]
                            except KeyError:
                                char_genders[key] = [1 if i in male else 0 for i in value]
                for key, value in char_genders.items():
                    char_genders[key] = Counter(value).most_common(1)[0][0]
            f.write(json.dumps({"movie_id": movie_index, "char_genders": char_genders}))
            f.write('\n')
            if counter % 10 == 0:
                print(f"{counter} many movies done")

# %%
