# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:43:12 2017

@author: lenovo laptop
"""
from nltk.corpus import stopwords

def create_datafile(cfile, cfile2, rfile, dfile):
    cachedStopWords = set(stopwords.words("english"))
    with open(cfile, "r", encoding="utf8") as fc, open(cfile2, "r", encoding="utf8") as fc2,  open(dfile, "w", encoding="utf-8") as fd:
    	count = 1
    	for cl, cl2 in zip(fc, fc2):
            cit = ' '.join([word for word in cl.split() if word.lower() not in cachedStopWords])
            with open(rfile, "r", encoding="utf8") as fr:
                for rl in fr:
                    ref = ' '.join([word for word in rl.split() if word.lower() not in cachedStopWords])
                    if rl == cl2:
                        fd.write(cit.strip()+"\t"+ref.strip()+"\t" +str(1)+"\n")
                    else:
                        fd.write(cit.strip()+"\t"+ref.strip()+"\t" +str(0)+"\n")
            count += 1		
    fc.close()
    fr.close()
    fd.close()
