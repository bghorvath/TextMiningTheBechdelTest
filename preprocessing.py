# %%
## Importing packages

import csv
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
## Downloading imdb data

# !wget https://datasets.imdbws.com/title.basics.tsv.gz -P data/imdb_data
# !gzip -d imdb_data/title.basics.tsv.gz

# %%

imdb_variables = ['imdb_id','imdb_type', 'imdb_title', 'imdb_year', 'imdb_runtime', 'imdb_genre']

# %%
## Creating dictionary

imdb_dict = {}

with open('data/imdb_data/title.basics.tsv') as imdb_data:
    reader = csv.reader(imdb_data, delimiter='\t')
    
    for rows in reader:
        imdb_id = rows[0]
        imdb_type = rows[1]
        imdb_title = rows[2]
        imdb_year = rows[5]
        imdb_runtime = rows[7]
        try:
            imdb_genre = rows[8]
        except:
            imdb_genre = np.nan
        if imdb_id in imdb_dict:
            pass
        else:
            imdb_dict[imdb_id] = [imdb_type, imdb_title, imdb_year, imdb_runtime, imdb_genre]

# %%
## Defining functions for joining imdb data

def get_imdb_data(imdb_link, columns):
    
    variables = {}
    
    for i, column in enumerate(columns):
        try:
            if i == 0:
                variables[column] = 'tt'+re.search(r'\d{7}', imdb_link).group(0)
            else:
                variables[column] = imdb_dict[variables['imdb_id']][i-1]
        except:
            variables[column] = np.nan

    return pd.Series([variables[i] for i in columns])

# %%
## Reading back pickle

open_file = open('pickles/movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

# %%
## Adding imdb columns

movies_df[imdb_variables] = movies_df.apply(lambda x: get_imdb_data(x.imdb_link, imdb_variables), axis=1)

# %%

movies_df = movies_df[['imdb_id','imdb_title','imdb_type','imdb_year','imdb_runtime','imdb_genre','script_link']]

# %%
## Filtering for movies and by imdb_year availability and dropping duplicate rows

movies_df = movies_df[movies_df['imdb_type']=='movie']
movies_df = movies_df[movies_df['imdb_year'] != r'\N']
movies_df = movies_df.dropna(subset=['imdb_year'])
movies_df = movies_df.drop_duplicates()

# %%

movies_df = movies_df.reset_index(drop=True)

# %%
## Saving file into pickle

open_file = open('pickles/imdb_movies_df.pkl', 'wb')
pickle.dump(movies_df, open_file)
open_file.close()

# %%

plt.hist(movies_df['imdb_year'].sort_values(), bins = 50)

# %%
