# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:43:12 2017

@author: lenovo laptop
"""
from nltk.corpus import stopwords

def create_datafile(cfile, cfile2, dfile):
    #cachedStopWords = set(stopwords.words("english"))
    with open(cfile, "r", encoding="utf8") as fc, open(cfile2, "r", encoding="utf8") as fc2,  open(dfile, "w", encoding="utf-8") as fd:
    	count = 1
    	for cl, cl2 in zip(fc, fc2):
            #cit = ' '.join([word for word in cl.split() if word.lower() not in cachedStopWords])
            fd.write(cl.strip()+"\t"+ref.strip()+"\n")
            count += 1		
    fc.close()
    fc2.close()
    fd.close()
