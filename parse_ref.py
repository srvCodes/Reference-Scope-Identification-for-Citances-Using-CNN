# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:47:49 2017

@author: lenovo laptop
"""

from xml.dom import minidom
import re

def parse_ref(filepath):
    xmldoc = minidom.parse(filepath)
    print("reading")
    itemlist = xmldoc.getElementsByTagName('S')
    dest = re.sub("\.xml" , "", filepath)
    sid = []
    #print(len(itemlist))
    with open(dest+".ref_sent.txt", 'a', encoding="utf8") as f:
        for s in itemlist:
            sid.append(s.attributes['sid'].value)
            f.write(s.childNodes[0].nodeValue+"\n")
    f.close()
    return sid

def main():
    filepath = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Training-Set-2016/C94-2154_TRAIN/Reference_XML/C94-2154.xml"
    sid = parse_ref(filepath)
    print (sid)
if __name__ == "__main__":
  main()
