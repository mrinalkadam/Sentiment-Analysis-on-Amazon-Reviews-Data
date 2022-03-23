# !/usr/bin/env python
# coding: utf-8

# In[1]:


# import required libraries and methods from them

from platform import python_version

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import re

from bs4 import BeautifulSoup

import contractions

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


# check the python version being used by the jupyter notebook

python_version()

# ## Read Data

# In[3]:


# read the input dataset into a dataframe

df = pd.read_csv("data.tsv", sep='\t', quoting=3)

# ## Keep Reviews and Ratings

# In[5]:


# keep only reviews and ratings columns

df = df[["review_body", "star_rating"]]
df.sample(n=3, random_state=100)

# In[6]:


# find out the number of reviews falling under each distinct rating

df['star_rating'].value_counts()

# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[7]:


# check for null values in the reviews column

df['review_body'].isnull().sum()

# In[8]:


# check for null values in the ratings column

df['star_rating'].isnull().sum()

# In[9]:


# drop null value records from the dataframe

df.dropna(inplace=True)

# In[10]:


# find out the number of reviews falling under distinct ratings

print(df[((df['star_rating'] == 4.0) | (df['star_rating'] == 5.0))]['star_rating'].count(), ",",
      df[((df['star_rating'] == 1.0) | (df['star_rating'] == 2.0))]['star_rating'].count(), ",",
      df[df['star_rating'] == 3.0]['star_rating'].count())

# In[11]:


# label reviews falling under ratings 4 and 5 as 1, under ratings 1 and 2 as 0, and remove the rewiews with rating 3

df['class'] = np.where(((df['star_rating'] == 4.0) | (df['star_rating'] == 5.0)), 1, 0)
df = df[df['star_rating'] != 3.0]

# drop the rating column once you have the label('class') column

df.drop(['star_rating'], axis=1, inplace=True)

#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
#
#

# In[12]:


# select a total of 200000 reviews randomly with 100000 positive and 100000 negative reviews

# find out classes with label 1(positive) and with label 0(negative)

df_positive = df[df['class'] == 1]
df_negative = df[df['class'] == 0]

# select 100000 records randomly from both the positive and negative classes

df_positive = df_positive.sample(n=100000, random_state=100)
df_negative = df_negative.sample(n=100000, random_state=100)

# concat the above records together to get a sample of 200000 reviews consisting of 100000 random positive and 100000 random negative reviews

df = pd.concat([df_positive, df_negative]).reset_index()
df.drop(['index'], axis=1, inplace=True)

# # Data Cleaning

# In[13]:


# convert the reviews column to string type

df['review_body'] = df['review_body'].astype(str)

# find out the average length of the reviews in terms of character length in the dataset before cleaning

len_before_data_cleaning = df['review_body'].apply(len).mean()

# ## Convert the all reviews into the lower case.

# In[14]:


# convert the reviews column to lower case

df['review_body'] = df['review_body'].str.lower()

# In[15]:


# find out three sample reviews before (data cleaning + pre-processing)

df['review_body'].sample(n=3, random_state=100).values


# ## remove the HTML and URLs from the reviews

# In[16]:


# using BeautifulSoup, remove HTML tags from the reviews column

# function to remove HTML tags
def remove_html(string):
    # parse through html content
    bs = BeautifulSoup(string, "html.parser")

    for text in bs(['style', 'script']):
        # remove the tags
        text.decompose()

    # return data by retrieving the tag content
    return ' '.join(bs.stripped_strings)


# apply the remove_html function to the reviews column

df['review_body'] = df['review_body'].apply(lambda x: remove_html(x))


# In[17]:


# using RegEx, remove URLs from the reviews column

# function to remove URLS
def remove_url(string):
    result = re.sub(r'^https?:\/\/.*[\r\n]*', r' ', string, flags=re.MULTILINE)
    return result


# apply the remove_url function to the reviews column

df['review_body'] = df['review_body'].apply(lambda x: remove_url(x))

# ## remove non-alphabetical characters

# In[18]:


# using RegEx, remove the characters apart from alphabets and single apostrophe(required for contractions later) from the reviews column and replace them with a single space

df['review_body'] = df['review_body'].replace(r"[^a-zA-Z' ]\s?", " ", regex=True)

# replace the single apostrophe with no space

df['review_body'] = df['review_body'].replace("'", "", regex=True)

# ## Remove the extra spaces between the words

# In[19]:


# using RegEx, remove the extra spaces between words from the reviews column

df['review_body'] = df['review_body'].replace('\s+', ' ', regex=True)

# ## perform contractions on the reviews.

# In[20]:


# using the contractions library, perform contractions on the reviews

df['review_body'] = df['review_body'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df['review_body'] = [' '.join(map(str, d)) for d in df['review_body']]

# In[21]:


# find out the average length of the reviews in terms of character length in the dataset after cleaning

len_after_data_cleaning = df['review_body'].apply(len).mean()

# In[22]:


# print the average length of the reviews in terms of character length in the dataset before and after cleaning

print(len_before_data_cleaning, ",", len_after_data_cleaning)

# # Pre-processing

# In[23]:


# find out the average length of the reviews in terms of character length in the dataset before pre-processing

len_before_pre_processing = df['review_body'].apply(len).mean()

# ## remove the stop words

# In[24]:


# remove all general stop words from the reviews column

stop_words = stopwords.words('english')
df['review_body'] = df['review_body'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# ## perform lemmatization

# In[25]:


# perform lemmatization with POS tagging

whitespace_tokenizer = nltk.tokenize.WhitespaceTokenizer()
wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()


# funtion to return a POS form of a word
def pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dictionary = {"J": wordnet.ADJ,
                      "N": wordnet.NOUN,
                      "V": wordnet.VERB,
                      "R": wordnet.ADV}

    return tag_dictionary.get(pos_tag, wordnet.NOUN)


# function to lemmatize the text
def lemmatize_text(string):
    return [wordnet_lemmatizer.lemmatize(w, pos(w)) for w in whitespace_tokenizer.tokenize(string)]


df['review_body'] = df['review_body'].apply(lemmatize_text)
df['review_body'] = [' '.join(map(str, l)) for l in df['review_body']]

# In[26]:


# find out the three sample reviews after (data cleaning + pre-processing)

df['review_body'].sample(n=3, random_state=100).values

# In[27]:


# find out the average length of the reviews in terms of character length in the dataset after pre-processing

len_after_pre_processing = df['review_body'].apply(len).mean()

# In[28]:


# print the average length of the reviews in terms of character length in the dataset before and after pre-processing

print(len_before_pre_processing, ",", len_after_pre_processing)

# # TF-IDF Feature Extraction

# In[29]:


# transform the features into tf-idf features using TfidfVectorizer

vectorizer = TfidfVectorizer()

x = df['review_body']
y = df['class']

x_final = vectorizer.fit_transform(x)

# In[30]:


# Split the dataset into 80% training dataset and 20% testing dataset

x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.20, random_state=100)

# # Perceptron

# In[31]:


# train a Perceptron model on the training dataset

perceptron = Perceptron(n_jobs=-1, random_state=100)
perceptron.fit(x_train, y_train)

# In[32]:


# predict the labels of train values

y_train_pred = perceptron.predict(x_train)

# find the accuracy, precision, recall and f1_score of the Perceptron model on the training set

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# In[33]:


# predict the labels of test values

y_test_pred = perceptron.predict(x_test)

# find the accuracy, precision, recall and f1_score of the Perceptron model on the test set

print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))

# # SVM

# In[34]:


# standardize the features using StandardScaler

scalar = StandardScaler(with_mean=False)
x_train_std = scalar.fit_transform(x_train)
x_test_std = scalar.transform(x_test)

# train an SVM model on the training dataset

lin_svc = LinearSVC(random_state=100)
lin_svc.fit(x_train_std, y_train)

# In[35]:


# predict the labels of train values

y_train_pred = lin_svc.predict(x_train_std)

# find the accuracy, precision, recall and f1_score of the SVM model on the training set

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# In[36]:


# predict the labels of test values

y_test_pred = lin_svc.predict(x_test_std)

# find the accuracy, precision, recall and f1_score of the SVM model on the test set

print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))

# # Logistic Regression

# In[37]:


# train a Logistic Regression model on the training dataset

log_reg = LogisticRegression(n_jobs=-1, random_state=100)
log_reg.fit(x_train, y_train)

# In[38]:


# predict the labels of train values

y_train_pred = log_reg.predict(x_train)

# find the accuracy, precision, recall and f1_score of the Logistic Regression model on the training set

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# In[39]:


# predict the labels of test values

y_test_pred = log_reg.predict(x_test)

# find the accuracy, precision, recall and f1_score of the Logistic Regression model on the test set

print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))

# # Naive Bayes

# In[40]:


# train a Multinomial Naive Bayes model on the training dataset

multi_nb = MultinomialNB()
multi_nb.fit(x_train, y_train)

# In[41]:


# predict the labels of train values

y_train_pred = multi_nb.predict(x_train)

# find the accuracy, precision, recall and f1_score of the Multinomial Naive Bayes model on the training set

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# In[42]:


# predict the labels of test values

y_test_pred = multi_nb.predict(x_test)

# find the accuracy, precision, recall and f1_score of the Multinomial Naive Bayes model on the test set

print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))

# In[ ]:
