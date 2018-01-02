from __future__ import division
"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def tfidf_cosine(cit_file, ref_file):
    with open(cit_file, 'r', encoding="utf-8") as rf:
        train_set = rf.readlines()
    train_set = [x.strip() for x in train_set] 
    
    with open(ref_file, 'r', encoding="utf-8") as cf:
        test_set = cf.readlines()
    test_set = [x.strip() for x in test_set]

    train_set = ["The sky is blue.", "The sun is bright."] #Documents
    test_set = ["The sun in the sky is bright."] #Query
    
    train_tfidf = TfidfVectorizer().fit_transform(train_set)
    test_tfidf = TfidfVectorizer().fit_transform(test_set)

    cosine_similarities = linear_kernel(train_tfidf, test_tfidf).flatten()
    print (cosine_similarities)
    
def main():
    cit_file = "G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025/annotation/C02-1025.annv3.citance.txt"
    ref_file = "G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025/Reference_XML/C02-1025.ref_sent.txt"
    tfidf_cosine(cit_file, ref_file)
    
if __name__ == "__main__":
    main()"""

import string
import math
import os
import re

def tfidf_cosine(document_0, document_1):
    tokenize = lambda doc: doc.lower().split(" ")
    
    all_documents = [document_0, document_1]
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
    
    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
        for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
            skl_tfidf_comparisons.append(cosine_similarity(doc_0, doc_1))
        
    return sorted(skl_tfidf_comparisons)[0]
    
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude
'''
def main():
    path = "G:/NLP/scisumm-corpus-master/data/Training-Set-2016"
    for folder in os.listdir(path):
        #temp = re.sub("\_TRAIN" , "", folder)
        if os.path.isdir(os.path.join(path, folder)):
            file = os.path.join(path, folder)+".txt"
            file = file.replace('\\','/')
            dest = re.sub("\.txt" , "", file)
            with open(file, "r", encoding="utf8") as fs, open(dest+".tfidf.txt", "a", encoding="utf8") as fd:
                for line in fs:   
                    line = line.split("\t")
                    document_0 = line[0]
                    document_1 = line[1]
                    fd.write(str(tfidf_cosine(document_0, document_1))+"\n")
    
if __name__ == "__main__":
    main()
'''

document_0 = "The Earth revolves around the Sun."
document_1 = "The Moon revolves around the Earth."

print(tfidf_cosine(document_0, document_1))
