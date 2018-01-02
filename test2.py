# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:59:26 2017

@author: lenovo laptop
"""

from parse_ref import parse_ref
import os
import re 

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Training-Set-2016"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
    	dest = re.sub("\_TRAIN", "", folder)
    	file = os.path.join(path, folder)+"/Reference_XML/"+dest+".xml"
    	file = file.replace('\\','/')
    	parse_ref(file)
