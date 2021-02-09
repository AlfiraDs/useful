from bert_serving.client import BertClient
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
from tqdm import tqdm

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


data = pd.read_csv('/Users/aliaksandr.lashkov/Google Drive/data/train_E6oV3lV.csv').sample(frac=0.1, random_state=0)
# data.tweet = data.tweet.str.replace('[@#]', ' ')
# data.tweet = data.tweet.str.replace('/s?', ' ').str.strip()


# # remove old style retweet text "RT"
# tweet2 = re.sub(r'^RT[\s]+', '', tweet)
#
# # remove hyperlinks
# tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
#
# # remove hashtags
# # only removing the hash # sign from the word
# tweet2 = re.sub(r'#', '', tweet2)


# # instantiate tokenizer class
# tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
#                                reduce_len=True)
#
# # tokenize tweets
# tweet_tokens = tokenizer.tokenize(tweet2)



# stopwords_english = stopwords.words('english')
#
# print('Stop words\n')
# print(stopwords_english)
#
# print('\nPunctuation\n')
# print(string.punctuation)


# # Instantiate stemming class
# stemmer = PorterStemmer()
#
# # Create an empty list to store the stems
# tweets_stem = []
#
# for word in tweets_clean:
#     stem_word = stemmer.stem(word)  # stemming word
#     tweets_stem.append(stem_word)  # append to the list



X_train, X_test, y_train, y_test = train_test_split(data.tweet, data.label)

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(solver='lbfgs')),
])

model.fit(X_train, y_train)
print('tfidf f1:', f1_score(y_test, model.predict(X_test), average='binary'))


def get_embedding(text):
    bc = BertClient()
    return bc.encode([text]).ravel()


itr = tqdm(data.tweet.tolist(), desc='Creating embeddings')
embeddings = Parallel(n_jobs=10)(delayed(get_embedding)(tweet) for tweet in itr)

X_train, X_test, y_train, y_test = train_test_split(embeddings, data.label)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

print('bert f1:', f1_score(y_test, model.predict(X_test), average='binary'))
