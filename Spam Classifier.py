#!/usr/bin/env python
# coding: utf-8

# # <font color='red'> Spam SMS Classifier<font>
# ### Machine learning is so powerful that it is helping us in each possible area and saving precious time. One of the use cases of Machine learning is Spam SMS detection, Using Natural Language Processing we can easily classify Spam and non Spam SMS.
# # <font color='green'> Download data for spam detection<font>
# ## https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# # <font color='red'> Data Description : <font> 
# ### The collection is composed by just one text file, where each line has the correct class followed by the raw message. 
# ### Ham : Not spam and Spam : Spam
#  A subset of 5572 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available.
# # <font color='red'> Objective : <font> 
# ### Using machine learning model, our model should be able to classify Spam and normal SMS.   

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[2]:


sms_data = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t', names=['label','message'])
sms_data.head(5)


# ham means not spam and spam is spam SMS

# In[3]:


# Getting Size of data
sms_data.shape
print("We have 5572 SMS")


# In[4]:


# Check if data is balanced or not
sms_data['label'].value_counts()


# <font color='red'> This dataset is imbalanced , because number of sample of class ham is significantly larger than number of sample of class spam. But we have 747 spam, we will be able to classify SMS.<font>

# In[5]:


# making set of stopwords (Those words which do not play significant role in decision making of text processing)
stopwords = set(stopwords.words('english'))
print(stopwords)
print("There are ",len(stopwords)," in english")


# # <font color='green'> Text Preprocessing <font>
# <ul>
#   <li>Stemming : processing of removing suffix from words to bring into root form, ex: going,gone,goes changes to root word go.</li>
#   <li>Removing punctuations, take only alphabates.</li>
#   <li>Convert words into lowercase.</li>
#     <li>Remove stopwords because they don't play significant role.</li>
#     <li>Make coupus : list of documents</li>
# </ul>

# In[6]:


# intialize stemmer object
stemmer = PorterStemmer()
corpus = []
for i in range(len(sms_data)):
    # replace non-alphabates with space
    row_data = re.sub('[^a-zA-Z]'," ",sms_data['message'][i])
    # convert words into lowercase
    row_data = row_data.lower()
    # make list from sentences
    row_data_list = row_data.split()
    # remove stopwords
    important_row_data = [stemmer.stem(word) for word in row_data_list if word not in stopwords]
    data = ' '.join(important_row_data)
    # append to corpus
    corpus.append(data)


# # <font color='green'> Using "Bag of Words"  method <font>
# <ul>
#   <li>After text preprocessing , make differnt dimensions for differnt each words in corpus</li>
#   <li>If unique words in corpus is d and there are n documents,make a matrix of nxd.</li>
#   <li>Each sentence will  be represented by a d-dimension vectors.</li>
#     <li>Matrix will be sparse matrix</li>
# </ul>
#     

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X.shape)


# There are 5053 different words,but many of them have very less frequency , lets take only top 3000 most frequent words

# In[25]:


# let's take only top 2500 most frequent words, it will reduce time complexit
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print("Size of X : ",X.shape)
X[0:10]


# ## label have to classes , ham and spam. but machine should be given numeric value,let's convert it into numeric form, 0 ham and 1 for spam

# In[33]:


y = pd.get_dummies(sms_data['label'],drop_first=True)
print("size of y : ",y.shape)
print("Converted label :")
print(y[:5])
print('Original label :')
print(sms_data['label'][:5])


# # <font color='green'> Splitting data into train and test set <font>

# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)


# # <font color='green'> Using Naive Bayes classifier <font>

# In[51]:


from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train,y_train)


# # <font color='green'> Prediction <font>

# In[52]:


y_pred = spam_model.predict(X_test)


# # <font color='green'> Analyzing model <font>

# In[53]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[54]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Model achieved ",round(accuracy*100,3) ,"% accuracy.")


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[58]:


# for data prediction in during deployment
import pickle
filename = 'spam_detecter.pkl'
pickle.dump(spam_model, open(filename, 'wb'))






