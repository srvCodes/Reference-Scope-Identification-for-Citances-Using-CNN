# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:39:24 2017

@author: lenovo laptop
"""

from generate_test import create_datafile
import os

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Test-Set-2016"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        cfile = os.path.join(path, folder)+"/annotation/"+folder+".annv3.citance.txt"
        cfile2 = os.path.join(path, folder)+"/annotation/"+folder+".annv3.reference.txt"
        rfile = os.path.join(path, folder)+"/Reference_XML/"+folder+".ref_sent.txt"
        dfile = os.path.join(path, folder)+".txt"
        cfile = cfile.replace('\\','/')
        cfile2 = cfile2.replace('\\','/')
        dfile = dfile.replace('\\','/')
        create_datafile(cfile, cfile2, rfile, dfile)
