import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imageio import imread
from wordcloud import WordCloud
from functools import partial
sns.set(style='darkgrid', context='talk', palette='colorblind')


lemmatizer = WordNetLemmatizer()

excluded = ['via', 'towards', 'based', 'method', 'use', 'framework', 'task', 'learn', 'based',
            'model', 'network', 'neural', 'improve', 'deep', 'multi']

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(title):
    """lemmatize
    e.g., 'learning' -> 'learn'
    """
    word_list = nltk.word_tokenize(title)
    return [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w)) for w in word_list]


def remove_stopword(title):
    word_split = title
    valid_word = []
    for word in word_split:
        word = word.strip().strip(string.digits)
        if word != "":
            valid_word.append(word)
    word_split = valid_word
    stop_words = set(stopwords.words('english'))
    # add punctuations
    punctuations = list(string.punctuation)
    [stop_words.add(punc) for punc in punctuations]
    # remove null
    stop_words.add("null")
    stop_words.update(excluded)

    return [word for word in word_split if word not in stop_words]


def transform(title, stopword=True):
    title = title.strip()
    title = lemmatize(title)
    if stopword:
        title = remove_stopword(title)
    return ' '.join(title)


class PaperNode:
    """PaperNode class for storing paper information
    """
    def __init__(self, paper_id, title, link, keyword_ls, rating=0, abstract=None):
        """__init__ function for PaperNode

        Args:
            paper_id (str): unique id for each paper
            title (str): title of the paper
            link (str): link to the paper
            keyword_ls (list of str): list of keywords
            abstract (str, optional): abstract of the paper. Defaults to None.
        """
        self.paper_id = paper_id
        self.title = title
        self.link = link
        self.keyword_ls = keyword_ls
        self.rating = rating # optional
        self.abstract = abstract # optional
        self.connectedTo = {}
        self.connected_components = []

    def __str__(self):
        return self.title

    def addNeighbor(self, neighbor_id, weight=0):
        """add neighbor to the paper

        Args:
            neighbor_id (str): neighbor id
            weight (int, optional): weight of the edge. Defaults to 0.
        """
        self.connectedTo[neighbor_id] = weight

    def calWeight(self, another_paper_node):
        """calculate the weight between two papers

        Args:
            another_paper_node (PaperNode): another paper node

        Returns:
            weight (int): weight between two papers
        """
        weight = 0
        for keyword in another_paper_node.keyword_ls:
            if keyword in self.keyword_ls:
                weight += 1
        return weight

class PaperGraph:
    """PaperGraph class for storing papers and their connections
    """
    def __init__(self):
        self.paperDict = {}
        self.numPapers = 0

    def addPaper(self, paper_id, paper: PaperNode):
        """add paper to the graph

        Args:
            paper_id (str): paper id
            paper (PaperNode): paper node
        """
        self.numPapers += 1
        self.paperDict[paper_id] = paper

    def addPaperDict(self, paper_dict: dict):
        """add paper dictionary to the graph

        Args:
            paper_dict (dict): paper dictionary
        """
        self.paperDict = paper_dict

    def getPaper(self, paper_id):
        """get paper by paper id

        Args:
            paper_id (str): paper id

        Returns:
            PaperNode: paper node
        """
        if paper_id in self.paperDict:
            return self.paperDict[paper_id]
        else:
            return None

    def addEdge(self, paper1: PaperNode, paper2: PaperNode, weight=0, threshold=1):
        """add edge between two papers

        Args:
            paper1 (PaperNode): paper node 1
            paper2 (PaperNode): paper node 2
            weight (int, optional): weight of the edge. Defaults to 0.
            threshold (int, optional): threshold of the weight, edge with weight below it will be ignored. Defaults to 1.
        """
        if weight >= threshold and paper1.paper_id != paper2.paper_id:
            self.paperDict[paper1.paper_id].addNeighbor(self.paperDict[paper2.paper_id].paper_id, weight)
            self.paperDict[paper2.paper_id].addNeighbor(self.paperDict[paper1.paper_id].paper_id, weight)

    def getPaperDict(self):
        """get the paper dictionary"""
        return self.paperDict

    def getConnection(self):
        """get the connection of the graph"""
        for paper_id in self.paperDict:
            # also include the weights
            print(self.paperDict[paper_id].paper_id, self.paperDict[paper_id].connectedTo)

    def dfs(self, paper_id, visited):
        """dfs function for finding connected components

        Args:
            paper_id (str): paper id
            visited (set): set of visited paper ids
        """
        visited.add(paper_id)
        for neighbor in self.paperDict[paper_id].connectedTo:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

    def findConnectedComponents(self):
        """find connected components in the graph

        Returns:
            connected_components (list of set): list of connected components
        """
        visited = set()
        connected_components = []

        for paper_id in self.paperDict:
            if paper_id not in visited:
                component = set()
                self.dfs(paper_id, component)
                connected_components.append(component)

        return connected_components



# load paper list and ratings
year = 2024
df_paper = pd.read_csv(f'paperlist_{year}.tsv', index_col=0, sep='\t')
# print('# papers:', len(df_paper))
# df_paper.head()
df_rating = pd.read_csv(f'ratings.tsv', index_col=0, sep='\t')
# print('# ratings:', len(df_rating))
# df_rating.head()
# merge paper and rating by paper id
df = pd.merge(df_paper, df_rating, on='paper_id')
# print('# merged:', len(df))
# df.head()
# extract papers with rating >= 6, which are highly possible to be accepted
df = df[df['rating'] >= 6]
# print('# filtered:', len(df))
# df.head()

def show_50_MOST_APPEARED_TITLE_KEYWORDS():
    words = pd.Series(
        ' '.join(df_paper['title'].dropna().apply(transform)).split(' ')
    ).str.strip()

    counts = words.value_counts().sort_values(ascending=True)

    plt.subplots(dpi=300)
    counts.iloc[-50:].plot.barh(figsize=(8, 12), fontsize=15)
    plt.title(f'img/50 MOST APPEARED TITLE KEYWORDS ({year})', loc='center', fontsize='25',
              fontweight='bold', color='black')
    plt.savefig(f'img/50_most_title_{year}.png', dpi=300, bbox_inches='tight')
    plt.show()


def show_50_MOST_APPEARED_KEYWORDS():
    words = pd.Series(
        ', '.join(df_paper['keywords'].dropna().apply(partial(transform, stopword=False))).lower().replace(' learn', ' learning').split(',')
    ).str.strip()

    counts = words.value_counts().sort_values(ascending=True)

    plt.subplots(dpi=300)
    counts.iloc[-50:].plot.barh(figsize=(8, 12), fontsize=15)
    plt.title(f'50 MOST APPEARED KEYWORDS ({year})', loc='center', fontsize='25',
              fontweight='bold', color='black')
    plt.savefig(f'img/50_most_keywords_{year}.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_Title_Word_Cloud_for_ICLR_2024():
    words = pd.Series(
        ' '.join(df_paper['title'].dropna().apply(transform)).split(' ')
    ).str.strip()
    logo = imread('img/logo-mask.png')
    wordcloud = WordCloud(background_color="white", max_words=2000, max_font_size=100,
               width=1280, height=640, random_state=0, mask=logo).generate(' '.join(words))

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(logo)
    plt.imshow(wordcloud, interpolation='bilinear', alpha=.75)
    plt.title(f'TITLE WORDCLOUD ({year})', loc='center', fontsize='25',
              fontweight='bold', color='black')
    plt.axis("off")
    plt.savefig(f'img/logo_wordcloud_title_{year}.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_Keyword_Word_Cloud_for_ICLR_2024():
    words = pd.Series(
        ', '.join(df_paper['keywords'].dropna().apply(partial(transform, stopword=False))).lower().replace(' learn', ' learning').split(',')
    ).str.strip()
    logo = imread('img/logo-mask.png')
    wordcloud = WordCloud(background_color="white", max_words=2000, max_font_size=100,
               width=1280, height=640, random_state=0, mask=logo).generate(' '.join(words))

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(logo)
    plt.imshow(wordcloud, interpolation='bilinear', alpha=.75)
    plt.title(f'KEYWORD WORDCLOUD ({year})', loc='center', fontsize='25',
              fontweight='bold', color='black')
    plt.axis("off")
    plt.savefig(f'img/logo_wordcloud_keywords_{year}.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_stats_1_degree():
    paperIDs = df.index.values
    paperDict = {}
    numPapers = len(df)
    paperGraph = PaperGraph()

    for id in paperIDs:
        title = df.loc[id].title
        link = df.loc[id].link
        keyword_ls = [keyword.strip() for keyword in df.loc[id].keywords.split(',')]
        rating = df.loc[id].rating
        paperDict[id] = PaperNode(id, title, link, keyword_ls, rating)
        paperGraph.addPaper(id, paperDict[id])

    # len(paperDict)
    # paperDict['eUgS9Ig8JG'].keyword_ls
    # print(paperGraph.numPapers == numPapers)
    # print(len(paperIDs))
    for i in tqdm(range(len(paperIDs))):
        for j in range(len(paperIDs)):
            paper1 = paperDict[paperIDs[i]]
            paper2 = paperDict[paperIDs[j]]
            weight = paper1.calWeight(paper2)
            paperGraph.addEdge(paper1, paper2, weight, threshold=1)

    paperGraph.connected_components_1_degree = paperGraph.findConnectedComponents()

    connected_components_degree_1 = paperGraph.connected_components_1_degree
    # find the largest and smallest connected component
    largest_component = max(connected_components_degree_1, key=len)
    smallest_component = min(connected_components_degree_1, key=len)
    average_component = sum([len(component) for component in connected_components_degree_1]) / len(connected_components_degree_1)
    print('largest component size:', len(largest_component))
    print('smallest component size:', len(smallest_component))
    print('average component size:', average_component)

def show_stats_2_degree():
    paperIDs = df.index.values
    paperDict_2 = {}
    numPapers_2 = len(df)
    paperGraph_2 = PaperGraph()

    for id in paperIDs:
        title = df.loc[id].title
        link = df.loc[id].link
        keyword_ls = [keyword.strip() for keyword in df.loc[id].keywords.split(',')]
        rating = df.loc[id].rating
        paperDict_2[id] = PaperNode(id, title, link, keyword_ls, rating)
        paperGraph_2.addPaper(id, paperDict_2[id])

    for i in tqdm(range(len(paperIDs))):
        for j in range(len(paperIDs)):
            paper1 = paperDict_2[paperIDs[i]]
            paper2 = paperDict_2[paperIDs[j]]
            weight = paper1.calWeight(paper2)
            paperGraph_2.addEdge(paper1, paper2, weight, threshold=2)

    paperGraph_2.connected_components_2_degree = paperGraph_2.findConnectedComponents()

    connected_components_degree_2 = paperGraph_2.connected_components_2_degree
    # find the largest and smallest connected component
    largest_component = max(connected_components_degree_2, key=len)
    smallest_component = min(connected_components_degree_2, key=len)
    average_component = sum([len(component) for component in connected_components_degree_2]) / len(connected_components_degree_2)
    print('largest component size:', len(largest_component))
    print('smallest component size:', len(smallest_component))
    print('average component size:', average_component)

def show_greatest_2_degree():
    paperIDs = df.index.values
    paperDict_2 = {}
    numPapers_2 = len(df)
    paperGraph_2 = PaperGraph()

    for id in paperIDs:
        title = df.loc[id].title
        link = df.loc[id].link
        keyword_ls = [keyword.strip() for keyword in df.loc[id].keywords.split(',')]
        rating = df.loc[id].rating
        paperDict_2[id] = PaperNode(id, title, link, keyword_ls, rating)
        paperGraph_2.addPaper(id, paperDict_2[id])

    for i in tqdm(range(len(paperIDs))):
        for j in range(len(paperIDs)):
            paper1 = paperDict_2[paperIDs[i]]
            paper2 = paperDict_2[paperIDs[j]]
            weight = paper1.calWeight(paper2)
            paperGraph_2.addEdge(paper1, paper2, weight, threshold=2)

    paperGraph_2.connected_components_2_degree = paperGraph_2.findConnectedComponents()
    connected_components_degree_2 = paperGraph_2.connected_components_2_degree
    largest_component = max(connected_components_degree_2, key=len)

    # show the largest component in a table
    largest_component_paper_ids = list(largest_component)
    largest_component_paper_ids.sort()
    # how to show it in the terminal?
    print(df.loc[largest_component_paper_ids])


if __name__ == "__main__":
    # want to do a terminal interaction
    # people can choose to
    # 1. See "50 MOST APPEARED TITLE KEYWORDS (2024)"
    # 2. See "Title Word Cloud for ICLR (2024)"
    # 3. See "50 MOST APPEARED KEYWORDS (2024)"
    # 4. See "Keyword Word Cloud for ICLR (2024)"
    # 5. See "stats of 1 degree connected components of ICLR (2024) submissions with rating >= 6"
    # 6. See "stats of 2 degree connected components of ICLR (2024) submissions with rating >= 6"
    # 7. See "the greatest 2 degree connected component of ICLR (2024) submissions with rating >= 6"
    # 8. Back to the main menu
    # 9. Exit
    while True:
        print("Choose an option:")
        print("1. See '50 MOST APPEARED TITLE KEYWORDS (2024)'")
        print("2. See 'Title Word Cloud for ICLR (2024)'")
        print("3. See '50 MOST APPEARED KEYWORDS (2024)'")
        print("4. See 'Keyword Word Cloud for ICLR (2024)'")
        print("5. See 'stats of 1 degree connected components of ICLR (2024) submissions with rating >= 6'")
        print("6. See 'stats of 2 degree connected components of ICLR (2024) submissions with rating >= 6'")
        print("7. See 'the greatest 2 degree connected component of ICLR (2024) submissions with rating >= 6'")
        print("8. Back to the main menu")
        print("9. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            show_50_MOST_APPEARED_TITLE_KEYWORDS()
        elif choice == "2":
            show_Title_Word_Cloud_for_ICLR_2024()
        elif choice == "3":
            show_50_MOST_APPEARED_KEYWORDS()
        elif choice == "4":
            show_Keyword_Word_Cloud_for_ICLR_2024()
        elif choice == "5":
            show_stats_1_degree()
        elif choice == "6":
            show_stats_2_degree()
        elif choice == "7":
            show_greatest_2_degree()
        elif choice == "8":
            continue
        elif choice == "9":
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 9.")
