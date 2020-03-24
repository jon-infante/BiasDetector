import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pickle
import random
import joblib

#Only using 5 percent of each dataframe
p = 0.05
df = pd.read_csv('env/datasets/cleaned_articles1.csv',header=None,skiprows=lambda i: 1>0 and random.random() > p)
df2 = pd.read_csv('env/datasets/cleaned_articles2.csv',header=None,skiprows=lambda i: 1>0 and random.random() > p)

df.columns = ['NaN', 'Unnamed', 'id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content', 'cleaned']
df2.columns = ['NaN', 'Unnamed', 'id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content', 'cleaned']

#First batch:
n_s_breitbart = df[df.publication == 'Breitbart']
n_s_times = df[df.publication == 'New York Times']
#Second batch:
n_s_atlantic = df2[df2.publication == 'Atlantic']
n_s_post = df2[df2.publication == 'New York Post']

#Combining each of the articles into a list
n_s = list(n_s_times.iloc[:,11].values) + list(n_s_atlantic.iloc[:,11].values) \
 + list(n_s_post.iloc[:,11].values) + list(n_s_breitbart.iloc[:,11].values)

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
bias_model_file = "bias_model.pkl"

with open(bias_model_file, 'wb') as file:
    joblib.dump(bias_model, file)

# vectorizer = model.named_steps['vect']
# chi = model.named_steps['chi']
# clf = model.named_steps['clf']

print("accuracy score: " + str(bias_model.score(X_test, y_test)))
#Left bias
print(bias_model.predict(['Politicians encourage hate crimes by discussing Brexit, and formally triggering the process will bring   attacks to “another level” MPs have been told. [Polish community leaders claimed the aspects of Britain that migrants like best, its “culture of tolerance” and “diversity” have been shattered since the Brexit vote because people have been emboldened to carry out verbal and physical attacks against migrants.  “Every statement and every political activity around Brexit negotiations brings a spike in inquiries to our organisation. We expect when article 50 is triggered it will bring another level of discontent,” Barbara Drozdowicz, of the East European Resource Centre (EERC) told the Commons Home Affairs Committee. Ms. Drozdowicz fingered criticism of mass migration as a driver of ‘hate’ in response to a question by fierce Remain campaigner Chuka Umunna. The Labour MP pointed to “Leave. EU, UKIP obviously” when he asked the EERC director whether she believes the “Leave campaign” and   figures “bear a responsibility” for   attacks. “The campaign was built on controlling migration. ‘Controlling migration’ is not a neutral term,” she told MPs. When Conservative David Burrowes noted that most examples of ‘hate’ are expressed online, community leaders admitted they’ve neither seen nor had reports of content on the internet that’s abusive to migrants from Europe. Tadeusz Stenzel, the chair of trustees at the Federation of Poles in Great Britain, suggested there might be a language barrier which prevents European migrants from being able to understand abusive posts and report them. MPs heard that European migrants began reporting ‘hate crimes’ in around April of last year as the referendum campaign got into gear, but that only after Britons voted to leave the European Union (EU) was the resource centre inundated with an “explosion of calls”. Ms. Drozdowicz said migrants from Eastern Europe reported being told to “go home” and hearing ‘casual hate speech’ on public transport and that the centre heard about violent incidents including a Polish child being beaten up at school. Chairman of the committee, Yvette Cooper, said the details of incidents that were talked about in the session were “appalling”. “Hate crime is appalling,   and should have no place in our country,” she said. Heading the government’s inquiry into ‘hate crime’ in November, Ms. Cooper accused the campaigns of both Leave, during the EU referendum, and Donald Trump, in the U. S. presidential race, of having incited hatred.']))
