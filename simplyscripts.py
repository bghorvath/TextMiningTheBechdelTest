# %%

import numpy as np
import pandas as pd
import requests
import re
import os
import sys
import codecs
import time
import pickle

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# %%

def get_page_links():

    link_all_scripts = 'https://www.simplyscripts.com/movie-screenplays.html'

    driver_path = "./chromedriver"
    driver = webdriver.Chrome(executable_path=driver_path)

    driver.get(link_all_scripts)

    try:
        page_links = []

        links = driver.find_elements_by_xpath("/html/body/div[@id='blog']/div[@id='wrapper']/div[@id='left']/ul[2]/a[@class='navl2']")
        for i in links:
            if re.search(r'^[a-zA-Z]$',i.text):
                page_links.append(i.get_attribute('href'))

    finally:
        driver.quit()

    unique_page_links = []

    for element in page_links:
        if element not in unique_page_links:
            unique_page_links.append(element)

    return unique_page_links

# %%

def get_movies(unique_page_links):

    driver_path = "./chromedriver"

    driver = webdriver.Chrome(executable_path=driver_path)

    movies = []

    for i, page_link in enumerate(unique_page_links):

        driver.get(page_link)

        table = driver.find_element_by_xpath( # 3+i
            "/html/body/div[@id='blog']/div[@id='wrapper']/div[@id='wrapperleft']/div[@id='mainros']/div[@id='movie_wide']/table[@border='0']/tbody") # /tr[3]

        rows = table.find_elements_by_xpath('tr')[2:]

        for row in rows:
            movie_name = ''
            movie_link = ''
            imdb_link = ''
            try:
                td1 = row.find_element_by_xpath('td[1]/a')
                movie_name = td1.text
                movie_link = td1.get_attribute('href')
            except:
                movie_name = np.nan
                movie_link = np.nan
            try:
                td5 = row.find_element_by_xpath('td[5]/a')
                imdb_link  = td5.get_attribute('href')
            except:
                imdb_link = np.nan
            if movie_name != np.nan and movie_link != np.nan and imdb_link != '':
                movies.append([movie_name, movie_link, imdb_link])

    driver.quit()

    return movies

# %%
## Running the scrapers

start_time = time.time()
unique_page_links = get_page_links()
movies = get_movies(unique_page_links)
movies_df = pd.DataFrame(movies, columns=['movie_name','script_link','imdb_link'])

print("The script scraped %a movies in %d seconds" % (len(movies), time.time() - start_time))

open_file = open('movies_df.pkl', 'wb')
pickle.dump(movies_df, open_file)
open_file.close()

# %%
