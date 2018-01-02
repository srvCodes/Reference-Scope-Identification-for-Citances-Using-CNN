# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:43:12 2017

@author: lenovo laptop
"""
import re
    
def read_facet(filepath):
    dest = re.sub("\.txt" , "", filepath)
    with open(filepath, "r", encoding="utf8") as fs, open(dest+".facet.txt", 'a', encoding="utf8") as fd:
    	count = 0
    	for line in fs:   
            #if line.startswith("Citance"):
            if count % 3 == 0:
                line.strip("\n")
                facet = line.split('|')
                facet = facet[9].strip()
                facet = facet[facet.find(": ")+1:].lstrip()
                if facet.startswith("["):
                    facet = facet[facet.find("[")+1:facet.rfind("]")]
                    facetlist = facet.split(",")
                    facet = ""
                    for f in facetlist:
                        f = f.replace("'", "")
                        facet = facet + f + "\t"
                fd.write(facet+"\n")
            count += 1              
    fs.close()
    fd.close()
    
def main():
    read_facet("G:/NLP/scisumm-corpus-master/data/Development-Set-Apr8/D10-1083/annotation/D10-1083.annv3.txt")
    
if __name__ == "__main__":
    main()