# %%
## Importing packages

import numpy as np
import pandas as pd
import requests
import re
import os
import sys
from pathlib import Path
import shutil
import codecs
import time
import pickle

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# %%
## Reading pickle

open_file = open('pickles/movies_df.pkl', "rb")
movies_df = pickle.load(open_file)

# %%
## Parser parameters

temp_dir = 'data/temp/'
scripts_dir = 'data/scripts/'

parser_formats = ['html','.htm']
download_formats = ['.txt','.pdf','.rtf','.doc']

# %%

def get_temp_file_name():

    temp_files = os.walk(temp_dir)

    file_name = [a for a in temp_files][0][2][0]

    file_path = temp_dir+file_name

    p = Path(file_path)
    file_suffix = p.suffix
    file_stem = p.stem
    
    return file_stem, file_suffix

# %%

def clear_dir(dir):
    for root, _, files in os.walk(dir):
        for f in files:
            os.remove(os.path.join(root, f))

# %%

def wget_file(i, script_link):
    try:
        os.system(f"wget -P {temp_dir} \"{script_link}\"")

        temp_file_stem, temp_file_suffix = get_temp_file_name()

        if temp_file_suffix.lower() in download_formats:
            file_move(i)
            result = 'success'
        else:
            clear_dir(temp_dir)
            result = 'fail'
    except Exception as e:
        result = 'fail'
        print(f"Failed to wget {script_link}: {e}")
    
    return result

# %%

def downloads_complete(driver):
    if not driver.current_url.startswith("chrome://downloads"):
        driver.get("chrome://downloads/")
    return driver.execute_script("""
        var items = document.querySelector('downloads-manager')
            .shadowRoot.getElementById('downloadsList').items;
        if (items.every(e => e.state === "COMPLETE"))
            return items.map(e => e.fileUrl || e.file_url);
        """)

# %%
def selenium_dl(i, script_link):
    
    driver_path = "other/chromedriver"
    options = webdriver.ChromeOptions()
    options.add_experimental_option('prefs', {
    "download.default_directory": temp_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
    })
    driver = webdriver.Chrome(options=options, executable_path=driver_path)

    try:
        driver.get(script_link)

        # waits for all the files to be completed and returns the paths
        download_wait = WebDriverWait(driver, 10, 1).until(downloads_complete)

        temp_file_stem, temp_file_suffix = get_temp_file_name()

        if temp_file_suffix in download_formats:
            file_move(i)
            result = 'success'
        else:
            clear_dir(temp_dir)
            result = 'fail'
    except Exception as e:
        clear_dir(temp_dir)
        result = 'fail'
        print(f"Failed to selenium_dl {script_link}: {e}")
    finally:
        driver.quit()
    
    return result

# %%

def parser(i, script_link):

    driver_path = "other/chromedriver"
    driver = webdriver.Chrome(executable_path=driver_path)

    try:
        driver.get(script_link)

        body = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, 'html/body'))
            )
        #driver.find_element_by_xpath("html/body")
        
        with open(f"{scripts_dir}/{i}.txt", "w") as text_file:
            text_file.write(body.text)

    except Exception as e:
        clear_dir(temp_dir)
        result = 'fail'
        print(f"Failed to parse {script_link}: {e}")

    finally:
        driver.quit()


# %%

def file_move(i):
    
    try:
        temp_files = os.walk(temp_dir)

        for temp_file in temp_files:
            file_name = temp_file[2][0]

        file_path = temp_dir+file_name

        p = Path(file_path)
        file_suffix = p.suffix

        new_path = scripts_dir+str(i)+file_suffix

        shutil.move(file_path, new_path)
        result = 'success'
    except Exception as e:
        result = 'fail'
        print(f"Failed to move {file_path}: {e}")
    
    return result

# %%
## Main script for downloading

log = []

for i, script_link in enumerate(movies_df['script_link']): # range(2150,2176):

    if i not in txtdir:
        print(f"Starting the download of {i}")
        if script_link[-4:].lower() in parser_formats:
            result = parser(i, script_link)
            if result == 'fail':
                log.append([i, script_link, 'parser', 'fail'])
            else:
                log.append([i, script_link, 'parser', 'success'])

        elif script_link[-4:].lower() in download_formats:
            result = wget_file(i, script_link)
            if result == 'fail':
                log.append([i, script_link, 'wget_file', 'fail'])
                result = selenium_dl(i, script_link)
                if result == 'fail':
                    log.append([i, script_link, 'selenium_dl', 'fail'])
                else:
                    log.append([i, script_link, 'selenium_dl', 'success'])
            else:
                log.append([i, script_link, 'wget_file', 'success'])

        elif re.search(r'beingcharliekaufman',script_link,re.IGNORECASE):
            result = selenium_dl(i, script_link)
            if result == 'fail':
                log.append([i, script_link, 'selenium_dl', 'fail'])
            else:
                log.append([i, script_link, 'selenium_dl', 'success'])

        elif re.search(r'sfy',script_link,re.IGNORECASE):
            result = parser(i, script_link)
            if result == 'fail':
                log.append([i, script_link, 'parser', 'fail'])
            else:
                log.append([i, script_link, 'parser', 'success'])

        else:
            log.append([i, script_link, 'other', 'left_out'])

log_df = pd.DataFrame(log, columns = ['i','script_link','method','result'])

# %%
## Writing log to pickles

open_file = open('log_df.pkl', 'wb')
pickle.dump(log_df, open_file)
open_file.close()
