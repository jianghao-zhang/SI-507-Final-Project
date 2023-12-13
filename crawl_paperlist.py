import os
import time
import argparse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default="2024", help="Year of OpenReview papers. (default: 2024)")
parser.add_argument('--pages', type=int, default=100, help='Number of pages on the website. (default: 100)')
args = parser.parse_args()

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get(f'https://openreview.net/group?id=ICLR.cc/{args.year}/Conference')

cond = EC.presence_of_element_located((By.XPATH, '//*[@class="submissions-list"]/nav/ul/li[13]/a'))
print(cond)
WebDriverWait(driver, 60).until(cond)

with open('paperlist.tsv', 'w', encoding='utf8') as f:
    f.write('\t'.join(['paper_id', 'title', 'link', 'keywords', 'abstract']) + '\n')

for page in tqdm(range(1, args.pages+1)):
    text = ''
    elems = driver.find_elements('xpath', '//*[@class="submissions-list"]/ul/li')
    for i, elem in enumerate(elems):
        try:
            # parse title
            title = elem.find_element('xpath', './/h4/a[1]')
            link = title.get_attribute('href')
            paper_id = link.split('=')[-1]
            title = title.text.strip().replace('\t', ' ').replace('\n', ' ')
            # show details
            elem.find_element('xpath', './/*[@class="collapse-widget "]/a').click()
            time.sleep(0.2)
            # parse keywords & abstract
            items = elem.find_elements('xpath', './/*[@class="note-content"]/div')
            keyword = ''.join([x.text for x in items if 'Keywords' in x.text])
            abstract = ''.join([x.text for x in items if 'Abstract' in x.text])
            keyword = keyword.strip().replace('\t', ' ').replace('\n', ' ').replace('Keywords: ', '')
            abstract = abstract.strip().replace('\t', ' ').replace('\n', ' ').replace('Abstract: ', '')
            text += paper_id + '\t' + title + '\t' + link + '\t' + keyword + '\t' + abstract + '\n'
        except Exception as e:
            print(f'page {page}, # {i}:', e)
            continue

    with open(f'paperlist_{args.year}.tsv', 'a', encoding='utf8') as f:
        f.write(text)

    # next page
    try:
        driver.find_element('xpath', '//*[@class="submissions-list"]/nav/ul/li[13]/a').click()
        time.sleep(3)  # NOTE: increase sleep time if needed
    except:
        print('no next page, exit.')
        break
