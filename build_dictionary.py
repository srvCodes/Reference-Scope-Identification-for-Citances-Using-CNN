import csv
import os
import pandas as pd 
import scipy.stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
'''
path1 = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Development-Set-Apr8"
path2 = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Test-Set-2016"

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+".csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        feature = [(findFeature(x,y),labels) for x,y,labels in zip(a,b,label)]
        featureSets = featureSets + feature

'''

a = "named entity recognizer (NER) useful many NLP applications information extraction, question answering, etc. own, NER also provide users looking person organization names quick information."
b = "automatically derived based correlation metric value used (Chieu Ng, 2002a)."

vectorizer  = CountVectorizer(lowercase=True, stop_words='english')

a = a.split()
X = vectorizer.fit_transform(a)

chi2score = chi2(X)