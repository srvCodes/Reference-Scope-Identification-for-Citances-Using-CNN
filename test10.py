from test1 import create_facetfile
import os

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Development-Set-Apr8"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        ffile = os.path.join(path, folder)+"/annotation/"+folder+".annv3.facet.txt"
        rfile = os.path.join(path, folder)+"/annotation/"+folder+".annv3.reference.txt"
        dfile = os.path.join(path, folder)+".facet.txt"
        ffile = ffile.replace('\\','/')
        rfile = rfile.replace('\\','/')
        dfile = dfile.replace('\\','/')
        create_facetfile(rfile, ffile, dfile)