# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:36:41 2017

@author: lenovo laptop
"""

import re
    
def read_citance(filepath):
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+".citance.txt", 'a', encoding="utf8") as fd: # open fd in append mode
        count = 0
        for line in fs:   
            if count % 3 == 0:
                line.strip("\n")
                citance = line.split('|')
                citance = citance[6].strip()
                citance = " ".join(re.findall(">(.*?)<", citance))           
                fd.write(citance+"\n")
            count += 1
    fs.close()
    fd.close()

#citance = citance[citance.find(">")+1:citance.rfind("<")]
