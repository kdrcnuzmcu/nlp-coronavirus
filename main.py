from textblob import TextBlob
from textblob import Word

from nltk.corpus import stopwords

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import decomposition
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import xgboost as xgb
import numpy as np
import textblob
import string
import re
import nltk

from keras.preprocessing import text
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras import optimizers

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 150)
pd.set_option("display.expand_frame_repr",  False)
nltk.download("stopwords")

train = pd.read_csv("Corona_NLP_train.csv", encoding="ISO-8859-1")
test = pd.read_csv("Corona_NLP_test.csv", encoding="ISO-8859-1")

train.head()
train.shape
test.shape

train.isnull().sum()
test.isnull().sum()
#region Preprocessing
train_copy = train.copy()
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if not x.startswith("http")))
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: " ".join(re.sub("[^A-Za-z@#]", "", x) for x in x.split()))
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].str.lower()
sw = stopwords.words("english")
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
rares = pd.Series(" ".join(train_copy["OriginalTweet"]).split()).value_counts()
rares = rares[rares == 1]
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in rares))
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train_copy["OriginalTweet"] = train_copy["OriginalTweet"].apply(lambda x: x.strip())

test_copy = test.copy()
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if not x.startswith("http")))
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].apply(lambda x: " ".join(re.sub("[^A-Za-z@#]", "", x) for x in x.split()))
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].str.lower()
sw = stopwords.words("english")
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
rares = pd.Series(" ".join(test_copy["OriginalTweet"]).split()).value_counts()
rares = rares[rares == 1]
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in rares))
test_copy["OriginalTweet"] = test_copy["OriginalTweet"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train_last = train_copy.copy()
train_last["Sentiment"].value_counts()
train_last = train_last[train_last["Sentiment"] != "Neutral"]
train_last = train_last.reset_index(drop=True)
#endregion

X = train_last["OriginalTweet"]
y = train_last["Sentiment"]
X.shape
y.shape
y = y.apply(lambda x: 0 if x == "Extremely Negative" else x)
y = y.apply(lambda x: 0 if x == "Negative" else x)
y = y.apply(lambda x: 1 if x == "Positive" else x)
y = y.apply(lambda x: 1 if x == "Extremely Positive" else x)
y.value_counts()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)
X_train.shape
y_train.shape

X_test.shape
y_test.shape

# LE = preprocessing.LabelEncoder()
# y_train = LE.fit_transform(y_train)
# y_test = LE.fit_transform(y_test)

def vectorize(Vectorizer, train, test):
    Vectorizer.fit(train)
    return Vectorizer.transform(train), Vectorizer.transform(test)

CV = CountVectorizer()
X_train_cv, X_test_cv = vectorize(CV, X_train, X_test)

tfidf_Vec_word = TfidfVectorizer()
X_train_tfidf_Vec_word, X_test_tfidf_Vec_word = vectorize(tfidf_Vec_word, X_train, X_test)

tfidf_Vec_ngrams = TfidfVectorizer(ngram_range=(2, 3))
X_train_tfidf_Vec_ngrams, X_test_tfidf_Vec_ngrams = vectorize(tfidf_Vec_ngrams, X_train, X_test)

tfidf_Vec_chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
X_train_tfidf_Vec_chars, X_test_tfidf_Vec_chars = vectorize(tfidf_Vec_chars, X_train, X_test)

def fit_ang_cv(Estimator, X_train, X_test, y_train, y_test):
    model = Estimator.fit(X_train, y_train)
    accuracy = model_selection.cross_val_score(model, X_test, y_test, cv=10).mean()
    print("Dogruluk orani:", accuracy)

LR = linear_model.LogisticRegression()
fit_ang_cv(LR, X_train_cv, X_test_cv, y_train, y_test)

NB = naive_bayes.MultinomialNB()
fit_ang_cv(NB, X_train_cv, X_test_cv, y_train, y_test)

XGB = xgb.XGBClassifier()
fit_ang_cv(XGB, X_train_cv, X_test_cv, y_train, y_test)
