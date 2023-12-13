# Analysis of ICLR 2024 OpenReview Data

# Intro

This project is inspired by the [ICLR2021-OpenReviewData](https://github.com/evanzd/ICLR2021-OpenReviewData/tree/master) initiative. My aim is to crawl, visualize, and analyze data from the ICLR 2024 OpenReview platform. The goal is to provide a comprehensive understanding of the trends and dynamics in cutting-edge machine learning research. This analysis will help in identifying emerging topics, the overall direction of the field, and notable shifts in research focus.

# Data Source

The primary source of data will be the [ICLR 2024 OpenReview website](https://openreview.net/group?id=ICLR.cc/2024/Conference). I plan to extract detailed information about the submitted papers, focusing primarily on paper keywords, ratings, and final decisions. This data will offer insights into the most discussed topics, the quality of research, and the acceptance trends in the conference.

## Requirements

+ Install requirements
```bash
pip install argparse selenium pandas wordcloud nltk pandas imageio selenium tqdm
```

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
```


## Crawl Data

1. Run `crawl_paperlist.py` to crawl the list of papers (~0.5h).

```bash
python crawl_paperlist.py
```

2. Run `crawl_reviews.py` to crawl the reviews of papers (~0.5h).

```bash
python crawl_reviews.py
```

## Paper List & Ratings

The extracted paper list and corresponding ratings are as follows:
    + [paperlist_2024.tsv](./paperlist_2024.tsv) (2,401 submissions in total)
    + [ratings.tsv](./ratings.tsv) (2,401 submissions in total)


## Visualization
1. Run `visualization.ipynb` and  `build_keyword_graph.ipynb` to build the keyword graph and visualize it.

2. Run `python Interactive.py` to interact with the keyword graph on the CLI.

## Related projects

+ https://github.com/evanzd/ICLR2021-OpenReviewData
+ https://github.com/fedebotu/ICLR2022-OpenReviewData
+ https://github.com/EdisonLeeeee/ICLR2022-OpenReviewData
+ https://github.com/EdisonLeeeee/ICLR2023-OpenReviewData
