import csv
import os
import pandas as pd 
from dice_coeff import dice_recur
from cosine import get_cosine
from jaccard_sim import is_ci_token_stopword_set_match
from fuzz_string_matching import get_string_similarity
from rouge_score import calc_Rouge
from sequence_matcher import similar 
from surface_features import get_surface_features
from fmeasure import FMeasure
from word2vec_similarity import word2vec_score
from ner_overlap import get_NER_overlap_score
from find_numbers import get_number_similarity
from pmi import getsignificance
from wordnet_similarity import get_wordnet_based_similarity
from tfidf import tfidf_cosine
#from mine_POS_pats import POSFeatures

#path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Development-Set-Apr8"


path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Test-Set-2016"

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+".csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        with open("test.csv", "a", newline="") as f:
            cw = csv.writer(f)
            cw.writerows(a + '\t' + b + '\t' + label)

        
'''
# writing features to csv file
with open("test_no_Word2Vec_features.csv", "a", newline="") as f:
    cw = csv.writer(f)
    cw.writerows([v for _,v in sorted(d.items())] + [s] for d,s in featureSets)
'''


    

