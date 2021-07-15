# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:50:19 2021

@author: prana
"""

import pandas as pd 
messages = pd.read_csv("SMSSpamCollection",sep = '\t',
                       names = ['label','messages'])
messages.shape

 #cleaning and preprocessiong
import re
import nltk
#nltk.download(stopwords)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
word_lemmatize = WordNetLemmatizer()
ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['messages'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    
#creating the bag of words model
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

#training the model
from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size =0.2,random_state=0)


#Training the model using Naive Bayes
from sklearn.naive_bayes import MultinomialNB

spam_detect_classifier = MultinomialNB().fit(X_train,y_train)


y_pred = spam_detect_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix 
confusion_c = confusion_matrix(y_test,y_pred)

    
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


 #save the model in pickle file   
import pickle
file = open("NB_spam_model.pkl",'wb')
 
 # dump information to that file
pickle.dump(spam_detect_classifier,file)
  
  
model = open('NB_spam_model.pkl','rb')
get_model = pickle.load(model)
 

    
 
    
 
    
 