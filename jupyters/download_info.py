#!/usr/bin/env python
# coding: utf-8

## Shahnameh Characters

shahnameh_characters = 'https://fa.wikipedia.org/wiki/%D9%81%D9%87%D8%B1%D8%B3%D8%AA_%D8%B4%D8%AE%D8%B5%DB%8C%D8%AA%E2%80%8C%D9%87%D8%A7%DB%8C_%D8%B4%D8%A7%D9%87%D9%86%D8%A7%D9%85%D9%87'

import requests
import bs4

context = requests.get(shahnameh_characters).text
soup = bs4.BeautifulSoup(context, 'lxml') 

characters_urls = {
    token.get_text(): 'https://fa.wikipedia.org' + token['href'] for token in 
    soup.select('#mw-content-text > div.mw-parser-output > ul > li > a')
}

characters_information = []

label_selector = '#mw-content-text > div.mw-parser-output > table > tbody > tr > th.infobox-label'
data_selector = '#mw-content-text > div.mw-parser-output > table > tbody > tr > td.infobox-data'

for name, link in characters_urls.items():
    
    response = requests.get(link)
    if response.status_code != 200:
        characters_information.append({'name': name})
        continue
    
    sub_context = response.text
    sub_soup = bs4.BeautifulSoup(sub_context, 'lxml')
    
    labels, datas = [lab.get_text() for lab in sub_soup.select(label_selector)], \
    			[dat.get_text() for dat in sub_soup.select(data_selector)]
    
    if not (labels and datas):
        characters_information.append({'name': name})
        continue
    
    characters_information.append({'name': name, 'info': dict(zip(labels, datas))})

import pandas as pd

def update_informations():
    
    for row in characters_information:
        names = [row['name'].split('(')[0]]

        if 'info' in row:        
            for fi in fields:
                names.extend([x.strip() for x in row['info'].get(fi, '').split('،') if x])

        row.update({'regex': '|'.join(names)})
        
fields = [
    'نام',
    'لقب',
    'نام\u200cهای دیگر'
]

update_informations()

df = pd.DataFrame([
    {'name':row['name'], 'regex':row['regex']} for row in characters_information])

df.to_csv('../datasets/shahnameh_characters.csv', index=False)


## Shahnameh Cities

shahnameh_cities = 'https://fa.wikipedia.org/wiki/%D9%81%D9%87%D8%B1%D8%B3%D8%AA_%D8%AC%D8%A7%DB%8C%E2%80%8C%D9%87%D8%A7_%D8%AF%D8%B1_%D8%B4%D8%A7%D9%87%D9%86%D8%A7%D9%85%D9%87'

import requests
import bs4

context = requests.get(shahnameh_cities).text
soup = bs4.BeautifulSoup(context, 'lxml') 

cities = [
    token.get_text() for token in soup.select('#mw-content-text > div.mw-parser-output > ul > li > a')
]

cities = cities[:-3]
pd.DataFrame([{'city': x} for x in cities]).to_csv('../datasets/shahnameh_cities.csv', index=False)

