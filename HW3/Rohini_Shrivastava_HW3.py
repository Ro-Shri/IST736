# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:22:59 2021

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

files="C:\\Users\\shriv\\Downloads\\spooky-author-identification\\train\\train.csv"
filepath="C:\\Users\\shriv\\Downloads\\spooky-author-identification\\train\\train.csv"
"""
Reviews=[]   
Labels=[]    

with open(filepath,'r') as file:   
    file.readline() 
    for row in file:  
        print(row)
        NextLabel,NextReview=row.split(",", 1)
        Reviews.append(NextReview)
        Labels.append(NextLabel)
MovieCV=CountVectorizer(input='content',stop_words='english')

MovieVec=MovieCV.fit_transform(Reviews)
ColumnNames=MovieCV.get_feature_names()
MovieDF=pd.DataFrame(MovieVec.toarray(),columns=ColumnNames)
MovieDF.loc[:,'Total'] = MovieDF.sum(numeric_only=True, axis=1)
MovieDF.loc['Total']= MovieDF.sum(numeric_only=True, axis=0)

#sort the dataframe by least to greatest in word count to understand the data easier
MovieDF = MovieDF.sort_values(MovieDF.last_valid_index(), axis=1)

print("This is the label",Labels)
print("colnames", ColumnNames)
print("MovieVec",MovieVec)
"""
out="C:\\Users\\shriv\\Downloads\\spooky-author-identification\\train\\train_updated.csv"
filename="C:\\Users\\shriv\\Downloads\\spooky-author-identification\\train\\train_filename.csv"
file=open(files,"r")
new=open(filename,"w")
ToWrite="Label,Text\n"
new.write(ToWrite)
new.close()
new=open(filename, "a")
df=pd.DataFrame()
outfile=open(out,"w")

for row in file:
    Raw="The next row is: \n" + row +"\n"
    outfile.write(Raw) ## I am going to write this later again for comp
    row=row.lstrip()  ## strip all spaces from the left
    row=row.rstrip()  ## strip all spaces from the right
    row=row.strip()   ## strip all extra spaces in general
    Mylist=row.split(" ")
    print(Mylist)
    NewList=[]    
    for word in Mylist:
        inout = "The next word BEFORE is: " +  word + "\n"
        outfile.write(inout)
        word=word.lower()
        word=word.lstrip()
        word=word.strip("\n")
        word=word.strip("\\n")
        word=word.replace(",","")
        word=word.replace(" ","")
        word=word.replace("_","")
        word=re.sub('\+', ' ',word)
        word=re.sub('.*\+\n', '',word)
        word=re.sub('zz+', ' ',word)
        word=word.replace("\t","")
        word=word.replace(".","")
        word=word.replace("\'s","")
        word=word.strip()
        if word not in ["", "\\", '"', "'", "*", ":", ";"]:
            if len(word) >= 3:
                if not re.search(r'\d', word): ##remove digits
                    NewList.append(word)
                    inout = "The next word AFTER is: " +  word + "\n"
                    outfile.write(inout)
    inout = "\nThe label is: " +  label + "\n"
    outfile.write(inout)
    NewList.pop() ## removes last item
    Text=" ".join(NewList)
    Text=Text.replace("\\n","")
    Text=Text.strip("\\n")
    Text=Text.replace("\\'","")
    Text=Text.replace("\\","")
    Text=Text.replace('"',"")
    Text=Text.replace("'","")
    Text=Text.replace("s'","")
    #Text=re.sub(r"lower()
    Text=Text.lstrip()
    OriginalRow="ORIGINAL" + Raw
    outfile.write(OriginalRow)
    ToWrite=label+","+Text+"\n"
    new.write(ToWrite)
    outfile.write(ToWrite)
file.close()  
new.close()
outfile.close()
"""
df=pd.read_csv(filename)
df = df.dropna(how='any',axis=0)  ## axis 0 is rowwise

MyLabel = df["Label"]

DF_noLabel= df.drop(["Label"], axis=1)  #axis 1 is column

MyList=[]  #empty list
for i in range(0,len(DF_noLabel)):
    NextText=DF_noLabel.iloc[i,0]  ## what is this??
    MyList.append(NextText)
#print("heres my list", MyList[1:4])
Vect = CountVectorizer(input="filename", analyzer = 'word',stop_words="english")

CV = Vect.fit_transform(MyList)
colum=Vect.get_feature_names()
VectorizedDF=pd.DataFrame(CV.toarray(),columns=colum)

VectorizedDF.loc[:,'Total'] = VectorizedDF.sum(numeric_only=True, axis=1)
VectorizedDF.loc['Total']= VectorizedDF.sum(numeric_only=True, axis=0)

#sort the dataframe by least to greatest in word count to understand the data easier
VectorizedDF = VectorizedDF.sort_values(VectorizedDF.last_valid_index(), axis=1)

print(VectorizedDF)
print(MovieDF)
VectorizedDF.to_csv("C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW3\\dfout.csv")#output.close()
"""