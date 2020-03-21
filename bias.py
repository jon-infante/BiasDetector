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
n_s = list(n_s_breitbart.iloc[:,9].values) + list(n_s_post.iloc[:,9].values) \
 + list(n_s_atlantic.iloc[:,9].values) + list(n_s_times.iloc[:,9].values)

#Removing media names
n_s = [word.replace('New York Post','') for word in n_s]
n_s = [word.replace('Breitbart','') for word in n_s]
n_s = [word.replace('New York Times','') for word in n_s]
n_s = [word.replace('Atlantic','') for word in n_s]

#Adding biases for each media outlet
classes_Bias = np.asarray([1 for i in range(len(n_s_breitbart))] + \
[1 for i in range(len(n_s_post))] + [2 for i in range(len(n_s_atlantic))] + \
[2 for i in range(len(n_s_times))])

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

print(bias_model.predict(['With confirmation hearings beginning, attention is shifting slightly away from Trump and toward those who may make up his administration. Like the   under whom they may soon be serving, many of Trump’s nominees have been pulled from the upper echelons of the business community or are otherwise political outsiders. Also like Trump, many of them have come under significant scrutiny for how their pasts may lead to conflicts of interest with the roles they are nominated to serve. These are especially noteworthy because, unlike during the transitions to previous administrations, it appears that this year’s confirmation hearings will begin before the Office of Government Ethics has had adequate time to perform background checks. This highly unusual turn of events goes against the explicit objection of the Director of the OGE, who wrote in a letter to the Senate’s Democratic leadership that he is “not aware of any occasion in the four decades since OGE was established when the Senate has held a confirmation hearing before the nominee has completed the ethics review” and warned, “For as long as I am Director, OGE’s staff and agency ethics officials will not succumb to pressure to cut corners and ignore conflicts of interest. ” (Also of note is a letter from current Senate Majority Leader Mitch McConnell, who in 2008 sent   Majority Leader Harry Reid a letter detailing his expectations for thorough ethical clearance but is now questioning the motives of those calling for the same for Trump’s nominees.) Moreover, in contrast with the rules for the president, there are both explicit rules regarding conflicts of interest for appointed officials and a long history of appointed officials abusing their offices for personal financial gain. With regard to the former, the law “prohibits an executive branch employee from participating personally and substantially in a particular Government matter that will affect his own financial interests” or those of his family members, a general partner, or any organization with whom he or she “serves as an officer, director, trustee, general partner, or employee” or may one day be employed. Violations could result in jail time of up to five years, a fine of up to $50, 000, or both investigations into alleged misconduct fall under the purview of the Justice Department. For the latter, perhaps the most infamous example of   officeholders acting in their personal financial interests is the Teapot Dome Scandal of the 1920s, in which Warren G. Harding’s Secretary of the Interior Albert Fall was convicted of accepting bribes in administering the leases for oil reserves in Wyoming and California. The administration of Ulysses S. Grant, meanwhile, was plagued by scandals in seven federal departments, ranging from gold speculation in the State Department to bribery for postal contracts, to the Whiskey Ring scandal, which led to the resignation of Grant’s Supervisor of Internal Revenue and his personal secretary. In recent decades, more than 100 of Ronald Reagan’s appointees, including Attorney General Edwin Meese and Deputy Chief of Staff Michael Deaver, were investigated for financial impropriety, leading to numerous firings and resignations. During the Obama administration, opponents accused Steven Spinner, who served in multiple   roles in the Department of Energy, of inappropriately pushing for a $535 million loan guarantee for the solar company Solyndra while his wife worked at a law firm representing the company. Even more recently, allegations of conflicts of interest were arguably central to the 2016 presidential campaign, in which Trump repeatedly accused Hillary Clinton of inappropriately commingling her office as Secretary of State with her family’s foundation. Adding to all of these concerns is the fact that Trump’s family members appear likely to have a larger role in his administration than a president’s family has had in any administration since John F. Kennedy’s appointment of his brother as Attorney General led to the creation of     laws. Ivanka Trump, for example, is poised to take on many of the duties typically expected of a First Lady, and has sat in on at least one of her father’s meeting with a    her adult siblings, Donald Jr. and Eric, are members of their father’s transition team and often seen as among his closest confidants, although it remains unclear what, if any, official roles they will take on in his administration. And Trump’s    Jared Kushner, whom many expect will be one of the  ’s top advisers once he takes office, has   and publishing interests of his own that could intersect with his official duties. As my colleague Olga Khazan noted, the question with conflicts of interest is not if, but when and how much they will affect behavior. Research into doctors who receive gifts from drug companies —  often as apparently meaningless as a pen or clipboard —  shows that “even small kickbacks can change   individuals’ behavior” by subtly changing how a doctor perceives the company behind the gift. As such, The Atlantic will be expanding our coverage of Trump’s conflicts of interests to include those expected to serve in his administration. Below is the current list of Trump’s prospective administration members, official and otherwise, whose actions or financial entanglements have prompted concerns over conflicts of interest. Because the law requires most aspirants for   positions to resolve such questions before entering office, some have already taken the steps necessary to mitigate conflicts of interest these steps will be noted as applicable. The most recent updates appear at the top: Trump’s Family Members, Though Trump has not —  in fact, cannot —  appointed any members of his direct family to a post in his administration, it nevertheless appears that they will likely have significant input in his administration. According to numerous sources, Trump’s eldest daughter, Ivanka, is likely to take on a policy portfolio roughly on par with that of a typical first lady just what it will comprise is not yet known, although based on meetings she has taken since the election with Leonardo DiCaprio and Al Gore, climate change has arisen as a possibility. Meanwhile, according to NBC, her husband, Jared Kushner, will be serving as a senior adviser to the president. And though Trump’s adult sons, Donald Jr. and Eric, apparently will not be serving in any official capacity within their father’s administration, they are nonetheless widely seen as among his closest confidants and are members of his transition team. Already, Trump’s three aforementioned children’s proximity to the   has created significant conflicts of interest. (Trump’s other adult child, Tiffany, does not appear likely to be part of the administration.) Much of this revolves around their father’s professed commitment to resolving issues with his namesake corporation by putting his assets into a trust that will be managed by Donald Jr. and Eric. As I have previously written, though Trump and his supporters have referred to the plan he has described as a blind trust (or, more recently, a “  trust,” which is a thing that doesn’t exist) the children’s   role as advisers to their father means they don’t have nearly enough separation to maintain a true blind trust. (And since so much of Trump’s business derives from real estate and his personal brand, even a real blind trust would not be sufficient to allay concerns about conflicts of interest, as he will retain his knowledge of what those assets are). Additionally, since Trump’s election, Ivanka, Donald Jr. and Eric have all been photographed in meetings that compromise even the illusion of distance from their father’s political dealings. All three appeared in photos of a summit between their father and the leaders of numerous technology companies, as well as a meeting with the   and his Indian business partners. Ivanka and her husband sat in on a meeting between their father and the Japanese Prime Minister Shinzo Abe, even though Ivanka was at the time negotiating a branding deal with a company owned in part by the Japanese government.']))
