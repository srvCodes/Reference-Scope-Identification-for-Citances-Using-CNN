import pandas as pd 
import os
import csv
from nltk.corpus import stopwords
from create_datafile import create_datafile

def removed_stops(raw):
	words = raw.split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return(" ".join(meaningful_words))

def remove_stopwords(rfile, dfile):
	dataframe = pd.read_csv(rfile, header = None, delimiter = "\t")
	citances = dataframe[1]
	references = dataframe[2]
	total_num = citances.size
	cleaned_citances = []
	cleaned_references = []
	for i in range(0, total_num):
		cleaned_citances.append(removed_stops(citances[i]))
		cleaned_references.append(removed_stops(references[i]))
	fd = open(dfile, "w", encoding="utf-8") 
	out_csv = csv.writer(fd, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
	out_csv.writerows(dataframe[0]+"\t"+cleaned_citances+"\t"+cleaned_references+"\t"+str(dataframe[3])+"\n")
	fd.close()

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Development-Set-Apr8"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        rfile = os.path.join(path, folder)+".csv"
        dfile = os.path.join(path, folder)+"_modified.csv"
        rfile = rfile.replace('\\','/')
        dfile = dfile.replace('\\','/')
        remove_stopwords(rfile, dfile)




