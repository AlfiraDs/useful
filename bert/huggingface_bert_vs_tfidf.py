import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoTokenizer, pipeline, TFDistilBertModel


data = pd.read_csv('/content/drive/My Drive/data/train_E6oV3lV.csv').sample(frac=0.3)
X_train, X_test, y_train, y_test = train_test_split(data.tweet, data.label)

model = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words='english',
        # ngram_range=(1, 3)
        )),
    ('clf', SGDClassifier()),
])

model.fit(X_train, y_train)
print('tfidf f1:', f1_score(y_test, model.predict(X_test), average='binary'))


model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',
                                          # pad_to_max_length=True,
                                          # max_length=100000
                                          )
pipe = pipeline('feature-extraction', model=model,
                tokenizer=tokenizer)
features = pipe(X_train.to_list(),
                # pad_to_max_length=True
                )
print()


# model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# pipe = pipeline('feature-extraction', model=model,
#                 tokenizer=tokenizer)
# features = pipe('any text data or list of text data',
#                 pad_to_max_length=True)
# features = np.squeeze(features)
# features = features[:,0,:]
#
#
#
# data = fetch_20newsgroups(subset='train')
#
# model = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier()),
# ])
#
# model.fit(data.data, data.target)
# test = fetch_20newsgroups(subset='test')
# print('tfidf f1:', f1_score(test.target, model.predict(test.data), average='weighted'))
#
#
# # make a connection with the BERT server using it's ip address; do not give any ip if same computer
# bc = BertClient()
# # get the embedding
# # embedding = bc.encode(["I love data science and analytics vidhya."])
# embedding = bc.encode(data.data)
# model = SGDClassifier()
# model.fit(embedding, data.target)
#
# test_embeddings = bc.encode(test.data)
# print('bert f1:', f1_score(test.target, model.predict(test_embeddings), average='weighted'))
#
# # check the shape of embedding, it should be 1x768
# # print(embedding.shape)
