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

# !wget https://datasets.imdbws.com/title.basics.tsv.gz -P imdb_data
# !gzip -d imdb_data/title.basics.tsv.gz

# %%
## Creating dictionary

imdb_dict = {}

with open('imdb_data/title.basics.tsv') as imdb_data:
    reader = csv.reader(imdb_data, delimiter='\t')
    for rows in reader:
        key = rows[0]
        if key in imdb_dict:
            pass
        else:
            imdb_dict[rows[0]] = [rows[2],rows[1],rows[5]]

# %%
## Defining functions for joining imdb data

def get_imdb_data(imdb_link):
    try:
        imdb_id = 'tt'+re.search(r'\d{7}',imdb_link).group(0)
    except:
        imdb_id = np.nan
    try:
        imdb_title = imdb_dict[imdb_id][0]
    except:
        imdb_title = np.nan
    try:
        imdb_type = imdb_dict[imdb_id][1]
    except:
        imdb_type = np.nan
    try:
        imdb_year = imdb_dict[imdb_id][2]
    except:
        imdb_year = np.nan
    
    return pd.Series([imdb_id, imdb_title, imdb_type, imdb_year])

# %%
## Reading back pickle

open_file = open('movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

# %%
## Adding imdb columns

movies_df[['imdb_id','imdb_title','imdb_type','imdb_year']] = movies_df['imdb_link'].apply(get_imdb_data)

# %%

movies_df = movies_df[['imdb_id','imdb_title','imdb_type','imdb_year','script_link']]

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

open_file = open('movies_df.pkl', 'wb')
pickle.dump(movies_df, open_file)
open_file.close()

# %%

plt.hist(movies_df['imdb_year'].sort_values(), bins = 50)

# %%

movies_df.to_csv('movies_df.csv')

# %%
