# %%
## Importing packages

import numpy as np
import pandas as pd
import re
import os
import sys
import time
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# %%
## Reading pickles

input_dir = 'data/processed/'

# %%
## Reading filtered movies_df from pickle

open_file = open('pickles/fmovies_df.pkl', "rb")
fmovies_df = pickle.load(open_file)

# %%
## Reading bechdel_df

with open('pickles/bechdel.pkl','rb') as f:
    bechdel_df = pickle.load(f)

# %%
## Adding paragraph and dialogue count to primary dataframe

# Counting pgs and dgs

pg_dg_count = []

with open('data/movie_dialogues.txt','r') as f:
    for i, row in enumerate(f):
        movie_json = json.loads(row)
        movie_index = movie_json['movie_id']
        pg_len = len(movie_json['paragraphs'])
        dg_len = 0
        char_dg_len = 0
        for pg in movie_json['paragraphs']:
            dg_len = dg_len + len(pg['dialogues'])
            char_dg_len = char_dg_len + len([j for j in pg['dialogues'] if j['character'] != 'NA'])
        
        pg_dg_count.append([movie_index, pg_len, dg_len, char_dg_len])

# %%
# Creating df

pg_dg_count_df = pd.DataFrame(pg_dg_count, columns = ['movie_id','pg_count','dg_count','char_dg_count'])
pg_dg_count_df = pg_dg_count_df.set_index('movie_id', drop = True)

# %%
## Merging with primary movie df and bechdel_df

analysis_df = pd.DataFrame.merge(fmovies_df, pg_dg_count_df, how = 'inner', left_index=True, right_index=True)
analysis_df = pd.DataFrame.merge(analysis_df, bechdel_df, how='inner', left_index=True, right_index=True)

# %%
## Dropping duplicates: only keeping the script for a movie that had the most dialogues and paragraphs

analysis_df = analysis_df[analysis_df['dg_count'] == analysis_df.groupby('imdb_id')['dg_count'].transform('max')]
analysis_df = analysis_df[analysis_df['pg_count'] == analysis_df.groupby('imdb_id')['pg_count'].transform('max')]

# %%
## Looking at number of paragraphs

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(analysis_df['dg_count'].sort_values(ascending = True))

# %%
## Dropoff 1% lowest quantile

analysis_df = analysis_df[analysis_df['dg_count'] > analysis_df['dg_count'].quantile(0.01)]

# %%
## Saving df to pickle

with open('pickles/analysis.pkl','wb') as f:
    pickle.dump(analysis_df, f)

# %%
## Saving to csv for further analysis

analysis_df.to_csv('data/analysis.csv')

# %%
## Looking at movies per year

plt.hist(analysis_df['imdb_year'].sort_values(), bins = 100)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(analysis_df['imdb_year'].value_counts().sort_index())

# %%
## Save to csv for further analysis

analysis_df['imdb_year'].value_counts().sort_index().to_csv("data/hist.csv")

# %%

analysis_df_test = analysis_df.copy()[['imdb_year','not_count']]

# %%

analysis_df_test['not_count'] = (analysis_df_test['not_count'] > 0)#.apply(int)

# %%

analysis_df_test.groupby(['imdb_year','not_count']).size().to_csv('data/timeseries.csv')

# %%
