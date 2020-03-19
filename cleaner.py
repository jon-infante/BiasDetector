import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import time
from tqdm import tqdm_gui


#Progress bar for program
for i in tqdm_gui(range(100)):
    time.sleep(3)

df = pd.read_csv('env/datasets/articles1.csv')
df2 = pd.read_csv('env/datasets/articles2.csv')

stemmer = SnowballStemmer('english')
words = stopwords.words("english")

#Cleaning up article text
df['cleaned'] = df['content'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df2['cleaned'] = df2['content'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#Converting pandas dataframe to csv like file
df.to_pickle('cleaned_articles1.pkl')
df2.to_pickle('cleaned_articles2.pkl')
