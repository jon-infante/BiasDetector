import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


df = pd.read_csv('env/datasets/articles1.csv')
df2 = pd.read_csv('env/datasets/articles2.csv')
# Used to factor in repetitive words
words = stopwords.words('english')
# The process of breaking a word down to its root, prefixes and suffixes
stemmer = SnowballStemmer('english')

#Cleaning up article text to only alpha characters as well as lowercase
df['cleaned'] = df['content'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df2['cleaned'] = df2['content'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#Converting pandas dataframe to csv like file
df.to_csv('env/datasets/cleaned_articles1.csv')
df2.to_csv('env/datasets/cleaned_articles2.csv')
