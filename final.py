import json
import csv
import re
import tweepy
import nltk
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import sklearn.model_selection as model_selection
from sklearn.model_selection import cross_validate

#Different keys of Twitter developer account

consumer_key = "CONSUMER KEY"
consumer_secret = "CONSUMER SECRET" 
access_token = "ACCESS TOKEN" 
access_token_secret = "ACCESS TOKEN SECRET"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit = True)

#Data Extraction

def Extract_data(label,qu,n):
  df = pd.DataFrame()
  q = qu + ' -filter:retweets'
  for status in tweepy.Cursor(api.search, q=q, lang='en', tweet_mode='extended').items(n):
      df = df.append({'createdTime' : status.created_at, 'Tweet' : status.full_text.replace('\n',' '), 'User': status.user.screen_name.encode('utf-8'), 'label': label}, ignore_index=True)
  fname = qu + '.csv'
  df.to_csv(fname)
  return df

d2 = Extract_data('water', '@GHMCOnline AND water', 1000)
d3 = Extract_data('electricity', '@GHMCOnline AND (electricity OR electric)', 1000)
d4 = Extract_data('electricity', '@TsspdclCorporat AND power', 1000)
d5 = Extract_data('police', '(@HYDTP OR @TelanganaCOPs)', 1000)
d6 = Extract_data('solid waste', '(@GHMCOnline OR @KTRTRS) AND (garbage OR waste)', 1000)
d7 = Extract_data('sanitation', '(@GHMCOnline OR @KTRTRS) AND sanitation', 1000)
d8 = Extract_data('education', '(@GHMCOnline OR @KTRTRS) AND education', 1000)
d9 = Extract_data('environment', '(@GHMCOnline OR @KTRTRS) AND (environment OR trees OR plant OR tree OR pollution)', 1000)
d10 = Extract_data('economy', '(@GHMCOnline OR @KTRTRS OR @TelanganaCMO OR @trsharish) AND (economy OR finance OR financial OR gdp)', 1000)
d11 = Extract_data('fire', '(@GHMCOnline OR @KTRTRS OR @TelanganaCMO OR @TelanganaCMO OR @ysjagan) AND (fire OR emergency OR #DisasterResponse)', 1000)
d12 = Extract_data('health', '(@GHMCOnline OR @KTRTRS OR @TelanganaCMO) AND (health)', 1000)
d13 = Extract_data('health', '(@TelanganaHealth OR @Eatala_Rajender) AND (hospital OR medical OR health OR suffering)', 1000)
d14 = Extract_data('transport', '(@GHMCOnline OR @KTRTRS) AND (road OR bus OR transport OR train OR metro)', 1000)
d15 = Extract_data('telecommunication', '(@GHMCOnline OR @KTRTRS) AND (bsnl OR network OR communications OR 5G OR 4G OR fibre OR fiber OR broadband )', 1000)

df = pd.concat([d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15])

#Data Preprocessing

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as wt
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

def preProcessing(tweets):
  tweets = tweets.tolist()
  ps = PorterStemmer()
  processedTweets = []
  punct = ['.',',','?','!',';',':','-','_','(',')','{','}','[',']','&','*','+','=',"'",'"','<','>','|','\\','/','`','~','@','#']
  sWords = set(stopwords.words('english'))
  for tweet in tweets:
    tweet = tweet.encode().decode()
    tweet = tweet.encode('ascii', 'ignore').decode('ascii') #removing the emojis
    tweet = tweet.encode('latin-1', 'ignore').decode('latin-1')
    tweet = re.sub(r'http\S+', '', tweet) #removing the urls
    tweet = tweet.lower() #converting the text into lower cases
    words = wt(tweet) #tokenization
    processedWords = []
    for word in words:
      if word in punct or word in sWords:
        continue
      if words[words.index(word) - 1] != '@' and words[words.index(word) - 1] != '#':
        processedWords.append(word)
    processedWords.remove(processedWords[0])
    processedTweets.append(' '.join(processedWords))
  return processedTweets

df.Tweet = preProcessing(df.Tweet)
df1=df[['Tweet','label']].copy()
df1.columns=['tweet','label']
df1['category_id'] = df1['label'].factorize()[0]
category_id_df = df1[['label', 'category_id']].drop_duplicates()

category_to_id = dict(category_id_df.values)  # Dictionaries for future use
id_to_category = dict(category_id_df[['category_id', 'label']].values)
fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue']
df1.groupby('label').tweet.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF TWEETS IN EACH CATEGORY\n')
plt.xlabel('Number of ocurrences', fontsize = 10);

#CountVectorizer feature selection

vectorizer = CountVectorizer(ngram_range=(1, 3), 
                        stop_words='english')
features = vectorizer.fit_transform(df1.tweet).toarray()
labels = df1.category_id

print("Each of the %d tweets is represented by %d features" %(features.shape))

#Training the models

X = df1['tweet']
y = df1['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    KNeighborsClassifier(n_neighbors=1),
    ComplementNB(),
    DecisionTreeClassifier(random_state=0),
    SVC(),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
score=['f1_macro','f1_micro']
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_validate(model, features, labels, scoring=score, cv=CV)
    for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
        entries.append((model_name,f1_macro,f1_micro))
cv_df = pd.DataFrame(entries, columns=['model_name','f1_macro','f1_micro'])
f1_macro = cv_df.groupby('model_name').f1_macro.mean()
f1_micro = cv_df.groupby('model_name').f1_micro.mean()
acc = pd.concat([f1_macro,f1_micro], axis= 1, 
          ignore_index=True)
acc.columns = ['f1-macro_mean','f1-micro_mean']
acc=acc.sort_values(by=['f1-macro_mean'], ascending=False)
print(acc)   #Accuracy scores

#Plotting the metrics of 8 algorithms

plt.figure(figsize=(20,5))
sns.boxplot(x='model_name', y='f1_macro', 
            data=cv_df, 
            showmeans=True)
plt.title("f1-macro (cv = 5)\n", size=14);
plt.figure(figsize=(20,5))
sns.boxplot(x='model_name', y='f1_micro', 
            data=cv_df, 
            showmeans=True)
plt.title("f1-micro (cv = 5)\n", size=14);

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df1.index, test_size=0.2, 
                                                               random_state=1)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_test, y_pred, 
                                    target_names= df1['label'].unique()))

#Minimum Document Frequency method

final2=[]
x=1
while(x<=30):
    vectorizer = CountVectorizer(min_df=x,
                        ngram_range=(1, 3), 
                        stop_words='english',
                            )
    features = vectorizer.fit_transform(df1.tweet).toarray()
    labels = df1.category_id
    print("\nFor min-df=",x,", each of the %d tweets is represented by %d features\n" %(features.shape))
    
    X = df1['tweet'] # Collection of documents
    y = df1['label'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
    models = [
    LinearSVC(max_iter=100000),
    LogisticRegression(random_state=0,max_iter=10000),
    DecisionTreeClassifier(random_state=0),
    ]

    # 5 Cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))

    entries = []
    score=['f1_macro','f1_micro']
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, features, labels, scoring=score, cv=CV)
        i=0
        for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
            entries.append((i,model_name,f1_macro,f1_micro))
            i+=1
    cv_df = pd.DataFrame(entries, columns=['index','model_name','f1_macro','f1_micro'])
    f1_macro = cv_df.groupby('model_name').f1_macro.mean()
    f1_micro = cv_df.groupby('model_name').f1_micro.mean()
    demo_acc = pd.concat([f1_macro,f1_micro], axis= 1, ignore_index=True)
    demo_acc.columns = ['f1-macro_mean','f1-micro_mean']
    print("Scores for min-df=",x," are given below:\n")
    print(demo_acc)
    l1=demo_acc['f1-macro_mean'].tolist()
    l2=demo_acc['f1-micro_mean'].tolist()
    l3=[]
    l3=[x]+l1+l2
    final2.append(l3)
    x+=1
temp=pd.DataFrame(final2,columns=['mnf','DTC_f1-macro','LSVC_f1-macro','LR_f1-macro','DTC_f1-micro','LSVC_f1-micro','LR_f1-micro'])
temp.to_csv('scores from countvectorizer(mnf).csv')
print(temp)

#Plotting the results

temp.plot(x='min_df',y=['DTC_f1-macro','LSVC_f1-macro','LR_f1-macro'],marker='^',figsize=(10,5),ylim=(0.7,0.8));

#Maximum number of Features method

final1=[]
w=500
while(w<=20000):
    vectorizer = CountVectorizer(max_features=w,
                        ngram_range=(1, 3), 
                        stop_words='english',
                            )
    # We transform each complaint into a vector
    demo_features = vectorizer.fit_transform(df1.tweet).toarray()
    labels = df1.category_id
    print("\nFor max-featues=",w,", each of the %d tweets is represented by %d features\n" %(demo_features.shape))
    
    X = df1['tweet'] # Collection of documents
    y = df1['label'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
    models = [
    LinearSVC(max_iter=100000),
    LogisticRegression(random_state=0,max_iter=10000),
    DecisionTreeClassifier(random_state=0),
    ]

    # 5 Cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))

    entries = []
    score=['f1_macro','f1_micro']
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, demo_features, labels, scoring=score, cv=CV)
        i=0
        for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
            entries.append((i,model_name,f1_macro,f1_micro))
            i+=1
    cv_df = pd.DataFrame(entries, columns=['index','model_name','f1_macro','f1_micro'])
    f1_macro = cv_df.groupby('model_name').f1_macro.mean()
    f1_micro = cv_df.groupby('model_name').f1_micro.mean()
    demo_acc = pd.concat([f1_macro,f1_micro], axis= 1, ignore_index=True)
    demo_acc.columns = ['f1-macro_mean','f1-micro_mean']
    print("Scores for max features=",w," are given below:\n")
    print(demo_acc)
    l1=demo_acc['f1-macro_mean'].tolist()
    l2=demo_acc['f1-micro_mean'].tolist()
    l3=[]
    l3=[w]+l1+l2
    final1.append(l3)
    w+=500
temp1=pd.DataFrame(final1,columns=['mnf','DTC_f1-macro','LSVC_f1-macro','LR_f1-macro','DTC_f1-micro','LSVC_f1-micro','LR_f1-micro'])
temp1.to_csv('scores from countvectorizer(mnf).csv')
print(temp1)

#Plotting the results

temp1.plot(x='mnf',y=['DTC_f1-macro','LSVC_f1-macro','LR_f1-macro'],marker='^',figsize=(10,5),ylim=(0.7,0.8));

#TF-IDF Feature Selection
 
tfidf = TfidfVectorizer(sublinear_tf=True,
                        ngram_range=(1, 3), 
                        stop_words='english')

features1 = tfidf.fit_transform(df1.tweet).toarray()

labels = df1.category_id

print("Each of the %d tweets is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features1.shape))
X = df1['tweet'] 
y = df1['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    KNeighborsClassifier(n_neighbors=1),
    ComplementNB(),
    DecisionTreeClassifier(random_state=0),
    SVC(),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
score=['f1_macro','f1_micro']
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_validate(model, features1, labels, scoring=score, cv=CV)
    for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
        entries.append((model_name,f1_macro,f1_micro))
cv_df = pd.DataFrame(entries, columns=['model_name','f1_macro','f1_micro'])

f1_macro = cv_df.groupby('model_name').f1_macro.mean()
f1_micro = cv_df.groupby('model_name').f1_micro.mean()
acc1 = pd.concat([f1_macro,f1_micro], axis= 1, 
          ignore_index=True)
acc1.columns = ['f1-macro_mean','f1-micro_mean']
acc1=acc1.sort_values(by=['f1-macro_mean'], ascending=False)
print(acc1)

#Plotting the metrics of 8 algorithms

plt.figure(figsize=(20,5))
sns.boxplot(x='model_name', y='f1_macro', 
            data=cv_df, 
            showmeans=True)
plt.title("F1-MACRO MEAN (cv = 5)\n", size=14);
plt.figure(figsize=(20,5))
sns.boxplot(x='model_name', y='f1_micro', 
            data=cv_df, 
            showmeans=True)
plt.title("F1-MICRO MEAN (cv = 5)\n", size=14);

#Minimum number of features method

final3=[]
x=1
while(x<=30):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 min_df=x,
                                 ngram_range=(1, 3),
                                 stop_words='english',
                                 )
    
    demo_features1 = vectorizer.fit_transform(df1.tweet).toarray()
    labels = df1.category_id
    print("\nFor min-df=",x,", each of the %d tweets is represented by %d features\n" %(demo_features1.shape))
    
    X = df1['tweet'] 
    y = df1['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
    models = [
    LinearSVC(max_iter=100000),
    LogisticRegression(random_state=0,max_iter=10000),
    DecisionTreeClassifier(random_state=0),
    ]

    # 5 Cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))

    entries = []
    score=['f1_macro','f1_micro']
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, demo_features1, labels, scoring=score, cv=CV)
        i=0
        for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
            entries.append((i,model_name,f1_macro,f1_micro))
            i+=1
    cv_df = pd.DataFrame(entries, columns=['index','model_name','f1_macro','f1_micro'])
    f1_macro = cv_df.groupby('model_name').f1_macro.mean()
    f1_micro = cv_df.groupby('model_name').f1_micro.mean()
    demo_acc = pd.concat([f1_macro,f1_micro], axis= 1, ignore_index=True)
    demo_acc.columns = ['f1-macro_mean','f1-micro_mean']
    print("Scores for min-df=",x," are given below:\n")
    print(demo_acc)
    l1=demo_acc['f1-macro_mean'].tolist()
    l2=demo_acc['f1-micro_mean'].tolist()
    l3=[]
    l3=[x]+l1+l2
    final3.append(l3)
    x+=1
temp3=pd.DataFrame(final1,columns=['min_df','DTC_f1-macro','LSVC_f1-macro','LR_f1-macro','DTC_f1-micro','LSVC_f1-micro','LR_f1-micro'])
temp3.to_csv('scores from tf-idf.csv')
print(temp3)

#Plotting the results

temp3.plot(x='min_df',y=['DTC_f1-macro','LSVC_f1-macro','LR_f1-macro'],marker='^',figsize=(10,5),ylim=(0.7,0.8));

#Maximum number of features method

final4=[]
w=500
while(w<=20000):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  max_features=w,
                                  ngram_range=(1, 3),
                                  stop_words='english',
                                  )
    # We transform each complaint into a vector
    demo_features2 = vectorizer.fit_transform(df1.tweet).toarray()
    labels = df1.category_id
    print("\nFor max-featues=",w,", each of the %d tweets is represented by %d features\n" %(demo_features2.shape))
    
    X = df1['tweet'] 
    y = df1['label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)
    models = [
    LinearSVC(max_iter=100000),
    LogisticRegression(random_state=0,max_iter=10000),
    DecisionTreeClassifier(random_state=0),
    ]

    # 5 Cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))

    entries = []
    score=['f1_macro','f1_micro']
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_validate(model, demo_features2, labels, scoring=score, cv=CV)
        i=0
        for f1_macro,f1_micro in zip(accuracies['test_f1_macro'],accuracies['test_f1_micro']):
            entries.append((i,model_name,f1_macro,f1_micro))
            i+=1
    cv_df = pd.DataFrame(entries, columns=['index','model_name','f1_macro','f1_micro'])
    f1_macro = cv_df.groupby('model_name').f1_macro.mean()
    f1_micro = cv_df.groupby('model_name').f1_micro.mean()
    demo_acc = pd.concat([f1_macro,f1_micro], axis= 1, ignore_index=True)
    demo_acc.columns = ['f1-macro_mean','f1-micro_mean']
    print("Scores for max features=",w," are given below:\n")
    print(demo_acc)
    l1=demo_acc['f1-macro_mean'].tolist()
    l2=demo_acc['f1-micro_mean'].tolist()
    l3=[]
    l3=[w]+l1+l2
    final4.append(l3)
    w+=500
temp4=pd.DataFrame(final1,columns=['mnf','DTC_f1-macro','LSVC_f1-macro','LR_f1-macro','DTC_f1-micro','LSVC_f1-micro','LR_f1-micro'])
temp4.to_csv('scores from tf-idf(mnf).csv')
print(temp4)

#Plotting the results

temp4.plot(x='mnf',y=['DTC_f1-macro','LSVC_f1-macro','LR_f1-macro'],marker='^',figsize=(10,5));

#Classification of new texts

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True,
                        max_features=16000,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = DecisionTreeClassifier(random_state=0).fit(tfidf_vectorizer_vectors, y_train)

new_tweet = input('Enter any text to classify:')
print(model.predict(fitted_vectorizer.transform([new_tweet])))