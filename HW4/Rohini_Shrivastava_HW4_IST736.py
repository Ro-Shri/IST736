# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:46:51 2021

@author: shriv
"""
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import string
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
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
#print(df_senti)
df_lie = pd.read_csv(filenames, sep='\t')
df_lie.pop('sentiment')
#print(df_lie)

Vect = CountVectorizer(input="content", analyzer = 'word',stop_words="english")
MyVect_Bern=CountVectorizer(input='content', analyzer = 'word', stop_words='english', binary=True)
MyVect_IDF=TfidfVectorizer(input='content', analyzer = 'word', stop_words='english')

ReviewVect=Vect.fit_transform(Reviews)
Review_IDF=MyVect_IDF.fit_transform(Reviews)
ReviewBern=MyVect_Bern.fit_transform(Reviews)
#print(ReviewVect)
#print(Review_IDF)
#print(ReviewBern)

ColumnNames=Vect.get_feature_names()
ResDF=pd.DataFrame(ReviewVect.toarray(),columns=ColumnNames)
#print(ResDF)

ColumnNames1=MyVect_IDF.get_feature_names()
ColumnNames2=MyVect_Bern.get_feature_names()
#print("Column names: ", ColumnNames2)
#Create a name
ResIDF=pd.DataFrame(Review_IDF.toarray(),columns=ColumnNames1)
ResBern=pd.DataFrame(ReviewBern.toarray(),columns=ColumnNames2)
print(ResIDF)
print(ResBern)

frames=[ResDF,Labels_senti]
sent_df = ResDF
sent_df['label']=Labels_senti

framesIDF=[ResIDF,Labels_senti]
sent_idf = ResIDF
sent_idf['label']=Labels_senti

framesIDF=[ResBern,Labels_senti]
sent_Bern = ResBern
sent_Bern['label']=Labels_senti

print(sent_idf)

rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(sent_df, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(sent_idf, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(sent_Bern, test_size=0.3)

Test1Labels=TestDF1["label"]
Test2Labels=TestDF2["label"]
Test3Labels=TestDF3["label"]

TestDF1 = TestDF1.drop(["label"], axis=1)
TestDF2 = TestDF2.drop(["label"], axis=1)
TestDF3 = TestDF3.drop(["label"], axis=1)

Train1Labels=TrainDF1["label"]
Train2Labels=TrainDF2["label"]
Train3Labels=TrainDF3["label"]

TrainDF1 = TrainDF1.drop(["label"], axis=1)
TrainDF2 = TrainDF2.drop(["label"], axis=1)
TrainDF3 = TrainDF3.drop(["label"], axis=1)
print(TestDF1)


MyModelNB1= MultinomialNB()
MyModelNB2= MultinomialNB()
BernModel = BernoulliNB()

MyModelNB1.fit(TrainDF1, Train1Labels)
MyModelNB2.fit(TrainDF2, Train2Labels)
BernModel.fit(TrainDF3, Train3Labels)

Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
Prediction3 = BernModel.predict(TestDF3)

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
cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)

print("this is the prediction")
print(np.round(MyModelNB1.predict_proba(TestDF1),2))
print(np.round(MyModelNB2.predict_proba(TestDF2),2))
print(np.round(BernModel.predict_proba(TestDF3),2))

#print(TotalDF)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    #pretty print for confusion matrixes
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

print("\nThe matrix for DF senti is:")
print_cm(cnf_matrix1,['Neg','Pos'])
print("\nThe matrix for IDF senti is:")
print_cm(cnf_matrix2,['Neg','Pos'])
print("\nThe matrix for Bernoulli senti is:")
print_cm(cnf_matrix3,['Neg','Pos'])


#creating the code for the lie label now
lie_df = ResDF
lie_df['label']=Labels_lie
lie_idf = ResIDF
lie_idf['label']=Labels_lie
lie_bern = ResBern
lie_bern['label']=Labels_lie

print(lie_df)


rd.seed(1234)
TrainDF1L, TestDF1L = train_test_split(lie_df, test_size=0.3)
TrainDF2L, TestDF2L = train_test_split(lie_idf, test_size=0.3)
TrainDF3L, TestDF3L = train_test_split(lie_bern, test_size=0.3)

#print(TrainDF5)

Test1LabelsL=TestDF1L["label"]
Test2LabelsL=TestDF2L["label"]
Test3LabelsL=TestDF3L["label"]

TestDF1L = TestDF1L.drop(["label"], axis=1)
TestDF2L = TestDF2L.drop(["label"], axis=1)
TestDF3L = TestDF3L.drop(["label"], axis=1)

Train1LabelsL=TrainDF1L["label"]
Train2LabelsL=TrainDF2L["label"]
Train3LabelsL=TrainDF3L["label"]

TrainDF1L = TrainDF1L.drop(["label"], axis=1)
TrainDF2L = TrainDF2L.drop(["label"], axis=1)
TrainDF3L = TrainDF3L.drop(["label"], axis=1)

MyModelNB1L= MultinomialNB()
MyModelNB2L= MultinomialNB()
MyModelNB3L= BernoulliNB()


MyModelNB1L.fit(TrainDF1L, Train1LabelsL)
print("error 1")
MyModelNB2L.fit(TrainDF2L, Train2LabelsL)
print("error 2")
MyModelNB3L.fit(TrainDF3L, Train3LabelsL)

Prediction1L = MyModelNB1L.predict(TestDF1L)
Prediction2L = MyModelNB2L.predict(TestDF2L)
Prediction3L = MyModelNB3L.predict(TestDF3L)

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

cnf_matrix1L = confusion_matrix(Test1LabelsL, Prediction1L)
cnf_matrix2L = confusion_matrix(Test2LabelsL, Prediction2L)
cnf_matrix3L = confusion_matrix(Test3LabelsL, Prediction3L)

print("this is the prediction")
print(np.round(MyModelNB1L.predict_proba(TestDF1L),2))
print(np.round(MyModelNB2L.predict_proba(TestDF2L),2))
print(np.round(MyModelNB3L.predict_proba(TestDF3L),2))


print("\nThe matrix for DF lie is:")
print_cm(cnf_matrix1L,['True','False'])
print("\nThe matrix for IDF lie is:")
print_cm(cnf_matrix2L,['True','False'])
print("\nThe matrix for Bern lie is:")
print_cm(cnf_matrix3L,['True','False'])

TotalDF=ResDF
TotalDF.loc[:,'Total'] = TotalDF.sum(numeric_only=True, axis=1)
TotalDF.loc['Total']= TotalDF.sum(numeric_only=True, axis=0)

#sort the dataframe by least to greatest in word count to understand the data easier
TotalDF = TotalDF.sort_values(TotalDF.last_valid_index(), axis=1)

#from sklearn.svm import LinearSVC

SVM_Model=sklearn.svm.SVC(C=250, kernel='linear', degree=7, verbose=True, gamma="auto")
SVM_Model.fit(TrainDF1, Train1Labels)

SVM_matrix = confusion_matrix(Test1Labels, SVM_Model.predict(TestDF1))
print("The linear confusion matrix for the DF Sentiment is:")
print_cm(SVM_matrix,['True','False'])

SVM_Model2=sklearn.svm.SVC(C=200, kernel='rbf', degree=10, verbose=True, gamma="auto")

SVM_Model2.fit(TrainDF1, Train1Labels)
SVM_matrix2 = confusion_matrix(Test1Labels, SVM_Model2.predict(TestDF1))
print("The radial confusion matrix for the DF sentiment is:")
print_cm(SVM_matrix2,['True','False'])

SVM_Model3=sklearn.svm.SVC(C=200, kernel='poly',degree=2, gamma="auto", verbose=True)

SVM_Model3.fit(TrainDF1, Train1Labels)

SVM_matrix3 = confusion_matrix(Test1Labels, SVM_Model3.predict(TestDF1))
print("The polynomial confusion matrix for the DF sentiment is:")
print_cm(SVM_matrix3,['True','False'])


