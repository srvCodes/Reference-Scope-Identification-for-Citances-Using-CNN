# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:53:35 2017

@author: lenovo laptop
"""

"""import csv
with open("G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025.csv", 'r', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
            print (row)"""
            
from find_numbers import find_numbers
from jaccard_sim import is_ci_token_stopword_set_match
from dice_coeff import dice_recur
import os
import re

path = "G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8"
for folder in os.listdir(path):
    #temp = re.sub("\_TRAIN" , "", folder)
    if os.path.isdir(os.path.join(path, folder)):
        file = os.path.join(path, folder)+".txt"
        file = file.replace('\\','/')
        dest = re.sub("\.txt" , "", file)
        with open(file, "r", encoding="utf8") as fs, open(dest+"_num_overlap.txt", "a", encoding="utf8") as fd:
            for line in fs:   
                line = line.split("\t")
                n1 = find_numbers(line[0])
                n2 = find_numbers(line[1])
                s1 =  ' '.join(str(x) for x in n1)
                s2 =  ' '.join(str(x) for x in n2)
                fd.write(is_ci_token_stopword_set_match(s1,s2)+"\t"+dice_recur(s1,s2)+"\n")