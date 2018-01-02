# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:39:45 2017

@author: lenovo laptop
"""

import re
    
def read_citance(filepath):
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+".citance.txt", 'a', encoding="utf8") as fd:
        for line in fs:   
            if line.startswith("Citance"):
                line.strip("\n")
                citance = line.split('|')
                citance = citance[6].strip()
                citance = citance[citance.find(">")+1:citance.rfind("<")]
                fd.write(citance+"\n")
    fs.close()
    fd.close()