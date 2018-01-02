# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:43:12 2017

@author: lenovo laptop
"""
from nltk.corpus import stopwords

def create_datafile(cfile, cfile2, rfile, dfile):
    with open(cfile, "r", encoding="utf8") as fc, open(cfile2, "r", encoding="utf8") as fc2,  open(dfile, "w", encoding="utf-8") as fd:
        for cl, cl2 in zip(fc, fc2):
            cit_words = cl.split()
            refr_words = cl2.split()
            stops = set(stopwords.words("english"))
            citance_words = [w for w in cit_words if not w in stops]
            new_cl = " ".join( citance_words )
            ref_words = [w for w in refr_words if not w in stops]
            new_cl2 = " ".join(ref_words)
            with open(rfile, "r", encoding="utf8") as fr:
                for rl in fr:
                    ref_words = rl.split()
                    reference_words = [w for w in ref_words if not w in stops]
                    ref = " ".join( reference_words )
                    if ref == new_cl2:
                        fd.write(new_cl.strip()+"\t"+ref.strip()+"\t" +str(1)+"\n")
                    else:
                        fd.write(cl.strip()+"\t"+ref.strip()+"\t" +str(0)+"\n")				
    fc.close()
    fr.close()
    fd.close()
    
