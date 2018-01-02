# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:45:08 2017

@author: lenovo laptop
"""
"""import nltk

print('The nltk version is {}.'.format(nltk.__version__))"""

#dirlist = [ item for item in os.listdir(file) if os.path.isdir(os.path.join(root, item)) ]
#read_citance("G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/C02-1025/annotation/C02-1025.annv3.txt")

def create_facetfile(rfile, ffile, dest):
    with open(rfile, "r", encoding="utf8") as fr, open(ffile, "r", encoding="utf8") as ff, open(dest, 'a', encoding="utf8") as fd:
        for rl, fl in zip(fr, ff):
        	rl = rl.strip()
        	rl = rl.split(".")
        	fl = fl.strip()
        	fl = fl.split("\t")
        	for i in rl:
        		if len(i) is not 0:
        			for j in fl:
        				fd.write(i + "\t" + j + "\n")
    fr.close()
    ff.close()
    fd.close()