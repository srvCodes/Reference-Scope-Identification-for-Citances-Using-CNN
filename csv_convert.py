# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:46:39 2017

@author: lenovo laptop
"""

import csv
import os

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Facet-Test-Set-2016"
csv_file = "/home/saurav/Documents/nlp_intern/facet_train.csv"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        txt_file = os.path.join(path, folder)+".facet.txt"
        #csv_file = os.path.join(path2)+".facet.csv"
        txt_file = txt_file.replace('\\','/')
        #csv_file = csv_file.replace('\\','/')
        in_txt = csv.reader(open(txt_file, "r", encoding="utf-8"), delimiter = '\t')
        out_file = open(csv_file, 'a'   , encoding="utf-8")
        writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        #writer.writerow(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o"])
        
        #writer.writeheader()
        out_csv = csv.writer(out_file, delimiter = '\t', quoting = csv.QUOTE_MINIMAL)

        out_csv.writerows(in_txt)
        
        #writer = csv.DictWriter(out_file, fieldnames = ["Id", "Citances", "References"], delimiter = '\t')
        #writer.writeheader()
