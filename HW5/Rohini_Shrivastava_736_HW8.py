import requests
import json
import re
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

BaseURL="https://newsapi.org/v1/articles"

URLPost = {'apiKey': 'f0e6bbe6725d469a86388c5642b4e0bd',
                    'source': 'bbc-news', 
                    'pageSize': 100,
                    'sortBy' : 'top',
                    'totalRequests': 150}

#print(URLPost)

response1=requests.get(BaseURL, URLPost)
#print(response1)
jsontxt = response1.json()
print(jsontxt)
"""
MyFILE=open("News.csv","w")


WriteThis="Author,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

MyFILE=open("News.csv", "a")

for items in jsontxt["articles"]:
    print(items)             
    Author=items["author"]
    Title=items["title"]
    Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\ +', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\"', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'[^a-zA-Z]', " ", Title, flags=re.VERBOSE)
    Headline=items["description"]
    Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\ +', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
    WriteThis=Author+ "," + Title + "," + Headline + "\n"
    MyFILE.write(WriteThis)

MyFILE.close()

BBC_DF=pd.read_csv("News.csv", error_bad_lines=False)

print(BBC_DF.head())

for col in BBC_DF.columns: 
    print(col) 

print(BBC_DF["Headline"])

HeadlineLIST=[]

for nextH in BBC_DF["Headline"]:
    HeadlineLIST.append(nextH)

print("The headline list is")
print(HeadlineLIST)

MyCountV=CountVectorizer(input="content", lowercase=True, stop_words = "english")

MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
print(type(MyDTM))
vocab = MyCountV.get_feature_names()  # change to a list
MyDTM = MyDTM.toarray()  # convert to a regular array

print(list(vocab)[10:20])
ColumnNames=MyCountV.get_feature_names()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)


#MyDTM_DF.loc[:,'Total'] = MyDTM_DF.sum(numeric_only=True, axis=1)
#MyDTM_DF.loc['Total']= MyDTM_DF.sum(numeric_only=True, axis=0)


#sort the dataframe by least to greatest in word count to understand the data easier
#MyDTM_DF = MyDTM_DF.sort_values(MyDTM_DF.last_valid_index(), axis=1)

print(MyDTM_DF)

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

num_topics = 3

lda_model_DH = LatentDirichletAllocation(n_components=num_topics, max_iter=1000, learning_method='online')

LDA_DH_Model = lda_model_DH.fit_transform(MyDTM_DF)
print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)


#print("First headline...")

print(LDA_DH_Model[0])

print(LDA_DH_Model[1])

print(LDA_DH_Model[2])
print(LDA_DH_Model[3])
print(LDA_DH_Model[4])

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
        

print_topics(lda_model_DH, MyCountV, 15)


import matplotlib.pyplot as plt
import numpy as np

word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()
num_top_words = 10 ##
vocab_array = np.asarray(vocab)

fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

#plt.tight_layout()
plt.show()
"""