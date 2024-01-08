
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
pd.set_option('display.max_colwidth', None) #displays entire text in each row

# #check if the two categories(disaster tweets and non-disaster) are balances - yes, 4342:3271
# #print(train_df['target'].value_counts())

# # 1. TEXT NORMALIZATION
# #try changing all alphabets to lower case - no need for tf-idf automatically does this
# #train_df['text'] = train_df['text'].str.lower()
# #test_df['text'] = test_df['text'].str.lower()

# # 1a) try tokenization - no need for any vectorizers because built-in vectorizer do this automatically
# import nltk
# from nltk.tokenize import word_tokenize
# #train_df['text'] = train_df['text'].apply(word_tokenize)
# #print(train_df['text'])

# def tokenize(df, col):
#     if col not in df.columns:
#         raise ValueError(f'Column {column_name} not found in DataFrame')
    
#     df['col'] = df['col'].apply(word_tokenize)

# # 1b) try removing all punctuations - no need for CountVectorizer because it auto removes all special characters
# import string
# #train_df['text'] = train_df['text'].str.replace(f'[{string.punctuation}]', '', regex=True)
# #print(train_df['text'])

# def remove_punc(df, col):
#     if column_name not in df.columns: 
#         raise ValueError(f'Column {column_name} not found in DataFrame')
        
#     df['col'] = df['col'].str.replace(f'[{string.punctuation}]', '', regex=True)

# ## 1c) try stemming
# # from nltk.stem import PorterStemmer
# # stemmer = PorterStemmer()

# # def stem_text(token):
# #     return [stemmer.stem(word) for word in token]

# # train_df['text'] = train_df['text'].apply(stem_text)
# # print(train_df['text'])

#1d) try lemmatization
import spacy

nlp = spacy.load('en_core_web_sm')
def spacy_lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

#Apply the lemmatization function to your DataFrame
train_df['text'] = train_df['text'].apply(spacy_lemmatize_text)
test_df['text'] = test_df['text'].apply(spacy_lemmatize_text)

#2. VECTORIZATION 
# ## 2a) try CountVectorizer
# # count_vectorizer = feature_extraction.text.CountVectorizer()
# # train_vectors = count_vectorizer.fit_transform(train_df["text"])
# # test_vectors = count_vectorizer.transform(test_df["text"])

# # 2b) try TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
train_vectors = tfidf_vectorizer.fit_transform(train_df['text'])
test_vectors = tfidf_vectorizer.transform(test_df["text"])

# ## 1st attempt: [0.63366337 0.6122449  0.68407835]
# ## 2nd with punc removed: [0.63541092 0.59779759 0.6676876]
# ## 3rd with all lower cased: [0.63366337 0.6122449  0.68407835]
# ## 4th with punc removed & lower cased: [0.63541092 0.59779759 0.6676876 ]

# ## 3. Dimensionality Reduction 
# ## Checking dimensionality - #samples = 7613, #features = 21637 high dimensionality
# # sample_size, num_features = train_vectors.shape
# # print(sample_size, num_features)

# # # 3a) TruncatedSVD/LSA - model gets worse 
# # from sklearn.decomposition import TruncatedSVD

# # # Trying different numbers of components
# # n_components_range = range(1, 5)  # n_components = 2 should be good
# # explained_variances = []

# # for n_components in n_components_range:
# #     svd = TruncatedSVD(n_components=n_components)
# #     svd.fit(train_vectors)
# #     explained_variances.append(svd.explained_variance_ratio_.sum())

# # # Plotting
# # plt.plot(n_components_range, explained_variances)
# # plt.xlabel('Number of Components')
# # plt.ylabel('Cumulative Explained Variance')
# # plt.show()

# # n_components = 2 #is the number of dimensions to reduce the data to
# # svd = TruncatedSVD(n_components=n_components)
# # train_svd = svd.fit_transform(train_vectors)
# # test_svd = svd.transform(test_vectors)

# # ## 1st attempt (with just Truncated SVD): [0.5364851  0.52371342 0.62045889]

# # # 3b) LDA - model barely improves
# # #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# # n_components = 1  # For binary classification, the maximum is 1
# # lda = LDA(n_components=n_components)

# # train_lda = lda.fit_transform(train_svd, train_df['target'])
# # test_lda = lda.transform(test_svd)

# ## 1st attempt (with SVD and LDA): [0.53456221 0.54029851 0.62369668]

# ### 

# 4. Fitting the model 
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# ## 4a) Ridge Regression - best F1: 0.80049
# #clf = linear_model.RidgeClassifier()
# #scores = model_selection.cross_val_score(clf, train_vectors , train_df["target"], cv=3, scoring="f1")
# #print(scores)
# #clf.fit(train_vectors , train_df["target"])
# #sample_submission["target"] = clf.predict(test_vectors)

# ## 4b) Logistic Regression - best F1: 0.79436
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import f1_score, classification_report
# # from sklearn.model_selection import train_test_split

# # log_reg = LogisticRegression()
# # scores = model_selection.cross_val_score(log_reg, train_vectors, train_df["target"], cv=3, scoring="f1")
# # print(scores)
# # log_reg.fit(train_vectors, train_df['target'])
# # sample_submission["target"] = log_reg.predict(test_vectors)

# 4c) Support Vector Machine (SVM) - best F1: 0.8014
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

svm = SVC() 
scores = model_selection.cross_val_score(svm, train_vectors, train_df["target"], cv=3, scoring="f1")
print(scores)
svm_model = svm.fit(train_vectors, train_df['target'])
sample_submission["target"] = svm_model.predict(test_vectors)
sample_submission.to_csv("submission.csv", index=False)
