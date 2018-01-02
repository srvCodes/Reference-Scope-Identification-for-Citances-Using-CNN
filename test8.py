# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:41:58 2017

@author: lenovo laptop
"""

import re

def read_annt_sent(filepath):
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+".reference.txt", 'a', encoding="utf8") as fd:
        for line in fs:   
            if line.startswith("Citance"):
                line.strip("\n")
                annt_sent = line.split('|')
                sid = annt_sent[7].strip()
                sid = sid[sid.find("[")+1:sid.find("]")]
                sid = sid.replace("\'", "")
                annt_sent = annt_sent[8].strip()
                annt_sent = " ".join(re.findall(">(.*?)<", annt_sent))           
                fd.write(annt_sent+"\n")
    fs.close()
    fd.close()