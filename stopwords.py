# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 02:44:58 2017

@author: lenovo laptop
"""

import re
from nltk.corpus import stopwords
    
def read_file(filepath):
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+"_1.txt", 'a', encoding="utf8") as fd:
        for line in fs:   
            line.strip("\n")
            sent = line.split('\t')
            citance = sent[1]
            ref = sent[2]
            cit_words = citance.split()
            ref_words = ref.split()
            stops = set(stopwords.words("english"))
            citance_words = [w for w in cit_words if not w in stops]
            reference_words = [w for w in ref_words if not w in stops]
            citance = " ".join( citance_words )
            ref = " ".join( reference_words )
            fd.write(citance+"\t"+ref+"\n")
    fs.close()
    fd.close()
'''   
def main():
    read_file("G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025.txt")
    
if __name__ == "__main__":
    main()'''
