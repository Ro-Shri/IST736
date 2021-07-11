import nltk
import pandas as pd
import sklearn
import re
import os
import sys
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer

#show all the rows and columns so we can accurately determine the counts of words
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#two corpuses were created: negative reviews and positive reviews
negpath = "C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\\NEG\\"
pospath = "C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\\POS\\"

print(os.listdir(path))
file = os.listdir(path)

FileListPath = []
FileList = []
Dict={}
#MyColumns = []

#add all the negative reviews into a list
for name in os.listdir(negpath):
    filepath = negpath+ "\\"+name
    FileListPath.append(filepath)
    
    filename = name.split(".")
    FileList.append(filename[0])
    
#add all the positive reviews into a list
for name in os.listdir(pospath):
    filepath = pospath+ "\\"+name
    print(filepath)
    FileListPath.append(filepath)
    
    filename = name.split(".")
    FileList.append(filename[0])
    
#create a dictionary to update the indexes of the files
for i in range(0,len(FileList)):
    Dict[i]=FileList[i]
    
files = CountVectorizer(input = "filename", stop_words="english")##"content", )

#MyList=["C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\\NEG\\veronica.txt"] 

Vectorize = files.fit_transform(FileListPath)
print(type(Vectorize))
print(Vectorize)

MyColumns=files.get_feature_names()
print(MyColumns)
                        
df = pd.DataFrame(Vectorize.toarray(), columns=MyColumns)
df=df.rename(Dict, axis="index")

#create an additional row and column that count the total number of words. 
#the Total for row counts how many words were not a stop word in the review
#the Total for column looks at how many times the words showed up overall in all the reviews

df.loc[:,'Total'] = df.sum(numeric_only=True, axis=1)
df.loc['Total']= df.sum(numeric_only=True, axis=0)

#sort the dataframe by least to greatest in word count to understand the data easier
df = df.sort_values(df.last_valid_index(), axis=1)

#df.plot(kind="bar")
#output the dataframe for better legibility
#output = open("C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\\df.xls", "a")
df.to_csv("C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\\dfout.csv")#output.close()
display(df)
#
#last_row = df.values[-1].tolist()
#print(graph)
#df.plot(y='Total', use_index=True)
#Trying to make a wordcloud
#
#print(FileListPath)
#wordcl = pd.read_csv('C:\\Users\\shriv\\OneDrive\\Documents\\IST 736\\HW2\POS\\sloan.txt', sep=" ", header=None)
#print(wordcl)
