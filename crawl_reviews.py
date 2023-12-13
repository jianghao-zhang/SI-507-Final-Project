import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


parser = argparse.ArgumentParser()
parser.add_argument("--year", nargs="?", default="2024", help="Year of OpenReview papers. (default: 2024)")
args = parser.parse_args()

driver = webdriver.Chrome()

df = pd.read_csv(f'paperlist_{args.year}.tsv', sep='\t', index_col=0)

rating = dict()
ratings = dict()
# decisions = dict()
for paper_id, link in tqdm(list(df.link.items())):
    try:
        driver.get(link)
        xpath = '//div[@class="note-content"]/div[contains(., "Rating")]'
        cond = EC.presence_of_element_located((By.XPATH, xpath))
        WebDriverWait(driver, 60).until(cond)

        elems = driver.find_elements('xpath', xpath)
        assert len(elems), 'empty ratings'
        ratings[paper_id] = pd.Series([
            int(x.text.split(': ')[1]) for x in elems if x.text.startswith('Rating:')
        ], dtype=int)
        rating[paper_id] = ratings[paper_id].mean()
        # decision = [x.text.split(': ')[1] for x in elems if x.text.startswith('Decision:')]
        # decisions[paper_id] = decision[0] if decision else 'Unknown'
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(paper_id, e)
        ratings[paper_id] = pd.Series(dtype=int)
        rating[paper_id] = 0
        # decisions[paper_id] = 'Unknown'

df = pd.DataFrame(list(rating.items()), columns=['paper_id', 'rating'])
df = df.set_index('paper_id')
# df['decision'] = pd.Series(decisions)
# df.index.name = 'paper_id'
df.to_csv('ratings.tsv', sep='\t') # mean rating
