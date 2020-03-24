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

print(bias_model.predict(['CHARLESTON, S. C.  —   Seeming to abdicate one of his last chances to save his own life, the convicted killer Dylann S. Roof stood on Wednesday before the jurors who will decide his fate and offered no apology, no explanation and no remorse for massacring nine black churchgoers during a Bible study session in June 2015. Instead, in a strikingly brief opening statement in the sentencing phase of his federal death penalty trial, Mr. Roof repeatedly assured the jury that he was not mentally ill  —   undercutting one of the few mitigating factors that could work in his favor  —   and left it at that. “Other than the fact that I trust people that I shouldn’t and the fact that I’m probably better at constantly embarrassing myself than anyone who’s ever existed, there’s nothing wrong with me psychologically,” Mr. Roof, who is representing himself, told the jury, which found him guilty last month of the killings at Charleston’s Emanuel African Methodist Episcopal Church. Three minutes after walking to the lectern, Mr. Roof returned to the defense table, exhaling deeply. Any prospects for mercy by the jury had perhaps already been drained by the prosecution’s disclosure, in its opening statement, of a white supremacist manifesto written by Mr. Roof in the Charleston County jail sometime in the six weeks after his arrest. “I would like to make it crystal clear I do not regret what I did,” he wrote in his distinctive scrawl. “I am not sorry. I have not shed a tear for the innocent people I killed. ” Mr. Roof, who was then 21, continued: “I do feel sorry for the innocent white children forced to live in this sick country and I do feel sorry for the innocent white people that are killed daily at the hands of the lower race. I have shed a tear of   for myself. I feel pity that I had to do what I did in the first place. I feel pity that I had to give up my life because of a situation that should never have existed. ” As the government laid out its case for a death sentence, the prosecutor who read from the journal, Nathan S. Williams, an assistant United States attorney, told the jury of 10 women and two men that Mr. Roof’s deadly rampage was a premeditated act that had devastated the families of his victims. “The defendant didn’t stop after shooting one person or two or four or five he killed nine people,” Mr. Williams said, a few moments before he declared, “The death penalty is justified. ” Later, aided by a slide show, he described each of the victims and their lives, setting the stage for several days of   testimony by family members and friends of the victims. Mr. Williams emphasized that Mr. Roof was capable of remorse and regret, reminding jurors that he had left his mother a note of apology, but only for the pain he knew his actions would cause his own family. The presentations were a startling beginning to the trial’s sentencing phase, which is expected to run into next week in Federal District Court. On Dec. 15, after six days of testimony in which defense lawyers did not contest his guilt, the jury found Mr. Roof guilty of 33 counts, including hate crimes, obstruction of religion resulting in death, and firearms charges. Eighteen of those counts require the jury to decide whether to sentence Mr. Roof, now 22, to death or life in prison without the possibility of parole. To impose a death sentence, jurors must unanimously find that aggravating factors like premeditation and the number and vulnerability of the victims outweigh any mitigating factors, like the absence of prior violent behavior and demonstrations of redemption and remorse. Mr. Roof is also facing a death penalty trial in state court. Although many people in the courtroom had already heard Mr. Roof’s flat,   monotone during the guilt phase, when prosecutors played a video recording of his   confession to F. B. I. agents, his statement on Wednesday was his first to the jury. Mr. Roof chose to allow his   legal team to represent him during the guilt phase, but sidelined them during the penalty proceedings to prevent them from introducing any evidence regarding his family background or mental capacity. “The point is that I’m not going to lie to you, not by myself or through somebody else,” Mr. Roof told the jury. As his paternal grandparents watched from the second row on the left side of the courtroom, several women on the right side, which is reserved for victims’ family members, left their seats, one of them muttering curses. Mr. Roof has said he does not plan to call witnesses or present evidence on his behalf, and he did not   any of the prosecution’s witnesses on Wednesday. His approach stands in sharp contrast to the strategy of Justice Department lawyers, who have said they may call more than 30 witnesses, including at least one survivor of the attack, family members of the victims and federal law enforcement officials. Prosecutors began Wednesday with the widow of the Rev. Clementa C. Pinckney, the church’s slain pastor, and his two best friends. Jennifer Benjamin Pinckney, who was married to Mr. Pinckney for 15 years, narrated an affectionate and often lighthearted telling of their life together, illustrated by dozens of photographs of her husband  —   as a young saxophone player in a school band, attending the births of their two daughters, vacationing on Caribbean cruises and on a trip to Seattle. She described him as a   preacher who extended his ministry as “a voice for the voiceless” to his work in the South Carolina Legislature, where he served first in the House and then the Senate. Often exhausted by his dual roles, he was depicted in several pictures as having fallen asleep in the back seat of the family car and on a couch while reading to his daughters. “He was the person that I think every mom would be happy that her daughter would marry,” Ms. Pinckney, a school librarian, said. “He was that great catch. ” Ms. Pinckney also described her terror on the night of June 17, 2015, as she and the couple’s younger daughter, Malana, then 6, listened to the gunfire from their hiding place beneath a desk in her husband’s study. As her husband and the others were gunned down in the adjacent church fellowship hall, Ms. Pinckney struggled to keep her daughter quiet and still. “I was just like, ‘Shh, shh, shh,’ ” Ms. Pinckney said, “and I put my hand over her mouth, and she was holding on to me, and she put her hand on my mouth. ” “Mama, is Daddy going to die?” her daughter asked, Ms. Pinckney said. She said the hardest thing she had ever done was telling her two daughters early the next morning that their father had been killed. Ms. Pinckney said she had heard Mr. Roof try to open the door to the study, which she had locked when the shooting began. Another assistant United States attorney, Julius N. Richardson, asked why she thought she had been spared. “It wasn’t my time,” Ms. Pinckney answered. “I couldn’t see God taking both parents away from two small kids. ” The Rev. Kylon Middleton, an A. M. E. minister who had known Mr. Pinckney from childhood, described his lifelong friend as immensely precocious (he began preaching at 13) and strategically ambitious (he aspired to be both a bishop and, perhaps, the state’s first   governor). In addition to Mr. Pinckney, the victims were the Rev. DePayne Middleton Doctor, 49 Cynthia Hurd, 54 Susie Jackson, 87 Ethel Lee Lance, 70 Tywanza Sanders, 26 the Rev. Daniel Lee Simmons Sr. 74 the Rev. Sharonda   45 and Myra Thompson, 59. Near the end of the day, Ms. Thompson’s widower, the Rev. Anthony B. Thompson, the vicar of a Reformed Episcopal Church here, told jurors about their   marriage, their anniversary date to a beach, his wife’s determined demeanor and her commitment to the historic congregation. In testimony that was mixed with laughter and tears, Mr. Thompson recounted their final day together as she prepared for the evening study of the Gospel of Mark. “She had her glow,” he said. “I mean, this smile on her face. She was radiant. I just kept looking at her. ” Word of a shooting came hours later, and Mr. Thompson rushed to the church. He demanded to know whether she had been injured or killed. He eventually found out. “My whole world was gone,” he said. “I literally did not know what to do. Everything I did was for her, and she was gone. What am I here for? If she’s gone, what am I here for?”']))
