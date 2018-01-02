# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:52:32 2017

@author: lenovo laptop
"""
from read_citance import read_citance
from read_annt import read_annt
from read_facet import read_facet 
import os

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Training-Set-2017"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        file = os.path.join(path, folder)+"/annotation/"+folder+".ann.txt"
        file = file.replace('\\','/')
        #read_citance(file)
        #read_annt(file)
        read_facet(file)
        
	
print("Success")
        
