import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
import spacy

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
pd.set_option('display.max_colwidth', None) #displays entire text in each row

nlp = spacy.load('en_core_web_sm')

def spacy_lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

train_df['text'] = train_df['text'].apply(spacy_lemmatize_text)
test_df['text'] = test_df['text'].apply(spacy_lemmatize_text)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), 
    ('svm', SVC())
])

pipeline.fit(train_df['text'], train_df['target'])
predictions = pipeline.predict(test_df['text'])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission['target'] = predictions
sample_submission.to_csv('submission.csv', index=False)
print(sample_submission['target'])
