# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:07:12 2021

@author: shriv
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:46:51 2021

@author: shriv
"""
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import string
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import numpy as np


filenames="C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW4\\deception_data_converted_final.tsv"
df = pd.read_csv(filenames, sep='\t')
Reviews=df['review']
#print(Reviews)
Labels_lie=df['lie']
#print(Labels_lie)
Labels_senti=df['sentiment']
#print(Labels_senti)
Reviews.str.replace('"',"")
Reviews.str.replace("\'","")
print(Reviews)

df_senti=pd.read_csv(filenames, sep='\t')
df_senti.pop('lie')
print(df_senti)
df_lie = pd.read_csv(filenames, sep='\t')
df_lie.pop('sentiment')
print(df_lie)

Vect = CountVectorizer(input="content", analyzer = 'word',stop_words="english")
ReviewVect=Vect.fit_transform(Reviews)
print(ReviewVect)
ColumnNames=Vect.get_feature_names()
ResDF=pd.DataFrame(ReviewVect.toarray(),columns=ColumnNames)
print(ResDF)

TotalDF=ResDF
TotalDF.loc[:,'Total'] = TotalDF.sum(numeric_only=True, axis=1)
TotalDF.loc['Total']= TotalDF.sum(numeric_only=True, axis=0)

#sort the dataframe by least to greatest in word count to understand the data easier
TotalDF = TotalDF.sort_values(TotalDF.last_valid_index(), axis=1)
#add the labels back
#create a train and test set with the label 5 times
# remove the label from the set
#run the NBM model
frames=[ResDF,Labels_senti]
sent_df = ResDF
sent_df['label']=Labels_senti

print(sent_df)


rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(sent_df, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(sent_df, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(sent_df, test_size=0.3)
TrainDF4, TestDF4 = train_test_split(sent_df, test_size=0.3)
TrainDF5, TestDF5 = train_test_split(sent_df, test_size=0.3)

print(TrainDF5)

Test1Labels=TestDF1["label"]
Test2Labels=TestDF2["label"]
Test3Labels=TestDF3["label"]
Test4Labels=TestDF4["label"]
Test5Labels=TestDF5["label"]

TestDF1 = TestDF1.drop(["label"], axis=1)
TestDF2 = TestDF2.drop(["label"], axis=1)
TestDF3 = TestDF3.drop(["label"], axis=1)
TestDF4 = TestDF4.drop(["label"], axis=1)
TestDF5 = TestDF5.drop(["label"], axis=1)

Train1Labels=TrainDF1["label"]
Train2Labels=TrainDF2["label"]
Train3Labels=TrainDF3["label"]
Train4Labels=TrainDF4["label"]
Train5Labels=TrainDF5["label"]

TrainDF1 = TrainDF1.drop(["label"], axis=1)
TrainDF2 = TrainDF2.drop(["label"], axis=1)
TrainDF3 = TrainDF3.drop(["label"], axis=1)
TrainDF4 = TrainDF4.drop(["label"], axis=1)
TrainDF5 = TrainDF5.drop(["label"], axis=1)

print(TestDF1)

MyModelNB1= MultinomialNB()
MyModelNB2= MultinomialNB()
MyModelNB3= MultinomialNB()
MyModelNB4= MultinomialNB()
MyModelNB5= MultinomialNB()

MyModelNB1.fit(TrainDF1, Train1Labels)
MyModelNB2.fit(TrainDF2, Train2Labels)
MyModelNB3.fit(TrainDF3, Train3Labels)
MyModelNB4.fit(TrainDF4, Train4Labels)
MyModelNB5.fit(TrainDF5, Train5Labels)

Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
Prediction3 = MyModelNB3.predict(TestDF3)
Prediction4 = MyModelNB4.predict(TestDF4)
Prediction5 = MyModelNB5.predict(TestDF5)

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)


print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)

cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)

print("\nThe confusion matrix is:")
print(cnf_matrix1)

cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

print("this is the prediction")
print(np.round(MyModelNB1.predict_proba(TestDF1),2))
print(np.round(MyModelNB2.predict_proba(TestDF2),2))
print(np.round(MyModelNB3.predict_proba(TestDF3),2))

