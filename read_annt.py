# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:14:44 2017

@author: lenovo laptop
"""
import re

def read_annt(filepath):
    print(filepath)
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+".reference.txt", 'a', encoding="utf8") as fd:
        count = 0
        for line in fs:   
            if count % 3 == 0:
                line.strip("\n")
                annt_sent = line.split('|')
                sid = annt_sent[7].strip()
                sid = sid[sid.find("[")+1:sid.find("]")] # extract ref id
                sid = sid.replace("\'", "") # remove "'"
                annt_sent = annt_sent[8].strip()
                annt_sent = " ".join(re.findall(">(.*?)<", annt_sent))  
                discourse = annt_sent[9].strip()
                discourse = discourse.replace(" ".join(re.findall("(.*?:)", discourse)), "")                       
                fd.write(annt_sent+"\t"+discourse+"\n")
            count += 1
    fs.close()
    fd.close()
