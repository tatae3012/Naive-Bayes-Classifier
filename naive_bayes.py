# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:46:56 2019

@author: VANSHIKA
"""
# Name: Vanshika Bansal
# Roll No: 1710110374
# I learned about Naive Bayes Theorem and python functions and libraries from the foolowing websites for the assignment:
# Stackoverflow, Towards Data Science, Stackexchange, DZone, scikit-learn.org, medium.com, www.kaggle.com

import csv
import os
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

#create a csv file v1 to store the emails
with open('v1.csv','a',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['class','message'])
file.close()
for i in range(1,11):
    for file in os.listdir('C:\\Users\\VANSHIKA\\Desktop\\bare\\'+str(i)+""):
    #Replace the location with the path of the lingspam dataset
        if file.endswith(".txt"):
            msg='C:\\Users\\VANSHIKA\\Downloads\\bare\\'+str(i)+"\\"+file
            f = open(msg,"r")
            txt=f.read()
            data=[file[0],txt]
        f.close()
        with open('v1.csv','a',newline='') as file:
             writer = csv.writer(file)
             writer.writerow(data)
        file.close()
with open('v1.csv','r') as file:
    read = csv.reader(file)
    lines = list(read)
for message in lines:
    if(message[0]=='s'):
        message[0]="spam"
    elif(message[0]!="class"):
        message[0]="ham"
    
with open('v1.csv','w', newline='') as writef:
    writer = csv.writer(writef)
    writer.writerows(lines)
file.close()
writef.close()

#Basic cleaning of unwanted symbols
def preparation(msg): 
    msg = msg.lower()
    msg = re.sub(r'http\S+', ' ', msg)
    msg = re.sub("\d+", " ", msg)
    msg = msg.replace('\n', ' ')
    return msg

#Removing of the stopwords
def stop_words(msg):
    stop = stopwords.words('english')
    msg = " ".join([word for word in msg.split() if word not in stop])
    return msg

#Lemmatising
def lemmatization(msg):
    lemmatizer = WordNetLemmatizer()
    msg = " ".join([lemmatizer.lemmatize(word, pos='v') for word in msg.split()])
    return msg

#Frequency pruning
def frequency_pruning(msg):
    from collections import Counter
    word_count = Counter()
    for row in msg :
        word_count.update(row.split())
    for word in word_count.copy():
        if word_count[word] <= 200:
            del word_count[word]
        if word_count[word] >= 400:
            del word_count[word]
    final = []
    for ele in word_count:
        final.append(ele)
    return final

df = pd.DataFrame(columns=['True Positives', 'False Positives', 'True Negatives', 'False Negatives', 'Accuracy', 'Precision', 'Recall',  'F1 Score'])

with open('v1.csv', 'r') as read:
           reader = csv.reader(read)
           lines = list(reader)
for message in lines:
    message[1]=preparation(message[1])
    message[1]=stop_words(message[1])
    message[1]=lemmatization(message[1])
    message[1]=lemmatization(message[1])
    
#Saving the cleaned data in v4.csv
with open('v4.csv', 'w', newline='') as write:
    writer = csv.writer(write)
    writer.writerows(lines)
write.close()
read.close()

ch = 1

file = pd.read_csv("v"+str(4)+".csv")

file = file[['class', 'message']]

#Dividing into training and testing data
start = 189
end = 480
accuracy_mean = 0
tnq = 0
fpq = 0
fnq = 0
tpq = 0 
aq = 0 
f1q = 0 
pq = 0 
rq = 0

x_train = pd.concat([file.iloc[1:start,1],file.iloc[end:,1]])
x_test = file.iloc[start:end,1]
y_train = pd.concat([file.iloc[1:start,0],file.iloc[end:,0]])
y_test = file.iloc[start:end,0]

print(y_train)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(file[:,0], file[:,1], test_size = 0.1, random_state = 0)

count_vect = TfidfVectorizer()
counts = count_vect.fit_transform(x_train.values)
classifier = MultinomialNB()
target = y_train.values
classifier.fit(counts, target)

test_count = count_vect.transform(x_test)
predictions = classifier.predict(test_count)
accuracy_mean = accuracy_mean+accuracy_score(y_test,predictions)

print(metrics.classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
a = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
p = precision_score(y_test, predictions, average="macro")
r = recall_score(y_test, predictions, average="macro") 
print(a,f1,p,r)
tp, fn, fp, tn = confusion_matrix(y_test, predictions)
df.loc[ch] = [tp, fp, tn, fn, a, p, r, f1]
tnq = tnq + tn 
fpq = fpq + fp 
fnq = fnq + fn 
tpq = tpq + tp
ch = ch + 1
aq = aq + a
f1q = f1q + f1
pq = pq + p
rq = rq + r
print("Performance of model : " + str(accuracy_mean))
print("\n")
df.loc[ch] = ["-","-","-","-","-","-","-","-","-"]
ch = ch + 1
df.loc[ch] = [tpq/9, fpq/9, tnq/9, fnq/9, aq/9, pq/9, rq/9, f1q/9]
ch = ch + 1
df.loc[ch] = ["-","-","-","-","-","-","-","-","-"]
ch = ch + 1

df.to_csv('result.csv')