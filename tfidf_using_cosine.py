from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:35:09 2017

@author: lenovo laptop
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
import pandas as pd 

def tfidf_cosine(train_set, test_set):
    '''
    with open(cit_file, 'r', encoding="utf-8") as rf:
        train_set = rf.readlines()
    train_set = [x.strip() for x in train_set] 
    
    with open(ref_file, 'r', encoding="utf-8") as cf:
        test_set = cf.readlines()
    test_set = [x.strip() for x in test_set]
    '''
    #train_set = ["The sky is blue.", "The sun is bright."] #Documents
    #test_set = ["The sun in the sky is bright."] #Query
    stopWords = stopwords.words('english')
    
    vectorizer = CountVectorizer(stop_words = stopWords)
    #print vectorizer
    transformer = TfidfTransformer()
    #print transformer
    
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    #print 'Fit Vectorizer to train set', trainVectorizerArray
    #print 'Transform Vectorizer to test set', testVectorizerArray
    denominator = (LA.norm(a)*LA.norm(b))
    if not denominator:
        return 0.0
    cx = lambda a, b : round(np.inner(a, b)/ denominator, 3)
    
    result_file = "/home/saurav/Documents/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025/tfidf_cosine.txt"
    with open(result_file, 'w', encoding="utf-8") as f:
        for vector in trainVectorizerArray:
            #print vector
            for testV in testVectorizerArray:
                #print testV
                cosine = cx(vector, testV)
                f.write(str(cosine)+"\t")
            f.write("\n")
    
    transformer.fit(trainVectorizerArray)
    #print
    #print transformer.transform(trainVectorizerArray).toarray()
    
    transformer.fit(testVectorizerArray)
    #print 
    tfidf = transformer.transform(testVectorizerArray)
    #print tfidf.todense()

def calc_tfidf(file):
    dataframe = pd.read_csv(file, header = None, delimiter = "\t")
    a = dataframe[1]
    b = dataframe[2]
    
    score = []
    for i,j in zip(a,b):
        score.append(tfidf_cosine(a,b))

    return score	

file = "/home/saurav/Documents/nlp_inten/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025.csv"
print(calc_tfidf(file))