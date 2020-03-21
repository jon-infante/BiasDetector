import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pickle


df = pd.read_pickle('env/datasets/cleaned_articles1.pkl')
df2 = pd.read_pickle('env/datasets/cleaned_articles2.pkl')

#First batch:
n_s_breitbart = df[df.publication == 'Breitbart']
n_s_times = df[df.publication == 'New York Times']
#Second batch:
n_s_atlantic = df2[df2.publication == 'Atlantic']
n_s_post = df2[df2.publication == 'New York Post']

#Combining each of the articles into a list
n_s = list(n_s_times.iloc[:,9].values) + list(n_s_atlantic.iloc[:,9].values) \
 + list(n_s_post.iloc[:,9].values) + list(n_s_breitbart.iloc[:,9].values)

#Removing media names
n_s = [word.replace('New York Times','') for word in n_s]
n_s = [word.replace('Atlantic','') for word in n_s]
n_s = [word.replace('New York Post','') for word in n_s]
n_s = [word.replace('Breitbart','') for word in n_s]

#Adding biases for each media outlet, 1 for left 2 for right
classes_Bias = np.asarray([1 for i in range(len(n_s_times))] + \
[1 for i in range(len(n_s_atlantic))] + [2 for i in range(len(n_s_post))] + \
[2 for i in range(len(n_s_breitbart))])

#Data to train the model
X_train, X_test, y_train, y_test = train_test_split(n_s, classes_Bias, test_size=0.2)

#Vectorizer takes the features and creates a matrix [bag of words], contains the
#amount of each word in the text. More common words will get a lower rating.
#SelectKBest selects the best features out of the matrix
#LinearSVC is the classifier we are using
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

bias_model = pipeline.fit(X_train, y_train)

#Saving model into a file
bias_model_file = "env/bias_model.pkl"

with open(bias_model_file, 'wb') as file:
    pickle.dump(bias_model, file)

# vectorizer = model.named_steps['vect']
# chi = model.named_steps['chi']
# clf = model.named_steps['clf']

print("accuracy score: " + str(bias_model.score(X_test, y_test)))
#Far right bias
print(bias_model.predict(['I love Donald Trump, building the border is the best thing we could have done. I also love capitalism too.']))
