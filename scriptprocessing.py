# %%
## Importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys
import time
import pickle
from pathlib import Path
import shutil
from striprtf.striprtf import rtf_to_text

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage

# %%
## Reading pickles and initializing directories

open_file = open('pickles/imdb_movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

scripts_dir = 'data/scripts/'
output_dir = 'data/processed/'

# %%
## Converting rtf to txt

for root, _, files in os.walk(scripts_dir):
    for f in files:
        
        file_path = os.path.join(root,f)

        p = Path(file_path)
        file_suffix = p.suffix.lower()
        file_stem = p.stem

        if file_suffix == '.rtf':
            with open(file_path, 'r') as f:
                rtf_text = f.read()
                text = rtf_to_text(rtf_text)
                with open(f"{output_dir}{file_stem}.txt", "w") as text_file:
                    text_file.write(text)

# %%
## Converting doc to txt with catdoc

for root, _, files in os.walk(scripts_dir):
    for f in files:
        
        file_path = os.path.join(root,f)

        p = Path(file_path)
        file_suffix = p.suffix.lower()
        file_stem = p.stem

        if file_suffix == '.doc':
            print(f)
            os.system(f"catdoc {os.path.join(root,f)} > {os.path.join(output_dir,file_stem+'.txt')}")

# %%
## Converting pdf to txt

fails = []

for root, _, files in os.walk(scripts_dir):
    for f in files:
        
        file_path = os.path.join(root, f)

        p = Path(file_path)
        file_suffix = p.suffix.lower()
        file_stem = p.stem

        if file_suffix == '.pdf':
            ## Decrypting copy-restricted pdfs
            try:
                os.system(f"qpdf --decrypt --password='' --replace-input {file_path}")
                print(f"Successfully decrypted {f}")
            except Exception as e:
                print(e)
            ## Parsing pdf
            try:
                resource_manager = PDFResourceManager(caching=True)
                out_text = StringIO()
                laParams = LAParams()
                text_converter = TextConverter(resource_manager, out_text, laparams=laParams)
                fp = open(file_path, 'rb')
                interpreter = PDFPageInterpreter(resource_manager, text_converter)

                for page in PDFPage.get_pages(fp, pagenos=set(), maxpages=0, \
                    caching=True, check_extractable=True):
                    interpreter.process_page(page)
                
                text = out_text.getvalue()
                fp.close()
                text_converter.close()
                out_text.close()

                ## Writing text to txt
                with open(os.path.join(output_dir, file_stem+".txt"), "w") as text_file:
                    text_file.write(text)
            except Exception as e:
                fails.append([file_stem, str(e)])

fails_df = pd.DataFrame(fails, columns = ['i','e'])

fails_df

# %%
## Moving txts to processed folder

for root, _, files in os.walk(scripts_dir):
    for f in files:
        
        file_path = os.path.join(root, f)

        p = Path(file_path)
        file_stem = p.stem
        file_suffix = p.suffix.lower()

        ## Needed to rename .TXT files to .txt
        new_path = os.path.join(output_dir, file_stem+'.txt')

        if file_suffix == '.txt':

            shutil.move(file_path, new_path)

# %%

## Looking at character count of the scripts
txtlen = {}

for root, _, files in os.walk(output_dir):
    for f in files:
        
        file_path = os.path.join(root,f)

        p = Path(file_path)
        file_suffix = p.suffix.lower()
        file_stem = p.stem

        if file_suffix == '.txt':
            with open(file_path, 'r', encoding = 'utf8', errors='ignore') as f:
                text = f.read().replace(" ","")
                txtlen[file_stem] = len(text)

txtlen_df = pd.DataFrame.from_dict(txtlen, orient='index', columns = ['len'])

# %%
## Filtering to only scripts that contain at least 10000 characters

txtlen_df_filtered = txtlen_df[txtlen_df.len > 10000]

plt.hist(txtlen_df_filtered['len'].values, bins = 100)

# %%
## Filtering to 90% mean

txtlen_df_90 = txtlen_df_filtered[(txtlen_df_filtered.len < txtlen_df_filtered.len.quantile(0.95)) \
    & (txtlen_df_filtered.len > txtlen_df_filtered.len.quantile(0.05))]

plt.hist(txtlen_df_90['len'].values, bins = 100)

# %%
## Merging movies_df with txtlen_df_90

txtlen_df_90['i'] = txtlen_df_90.index.astype(int)
movies_df['i'] = movies_df.index.astype(int)
fmovies_df = pd.DataFrame.merge(movies_df, txtlen_df_90, how = 'inner', left_on = 'i', right_on = 'i')

fmovies_df = fmovies_df.set_index('i')

fmovies_df = fmovies_df[['imdb_id','imdb_title','imdb_year','imdb_runtime','imdb_genre','len']]

# %%
## Writing filtered movies_df to pickle

open_file = open('pickles/fmovies_df.pkl', 'wb')
pickle.dump(fmovies_df, open_file)
open_file.close()

# %%