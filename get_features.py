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

def findFeature(i,j):
    dice_score = []
    cosine_score = []
    jaccard_score = []
    fuzzy_score = []
    sequence_matcher = []
    fmeasure = []
    word2vec_scores = []
    wordnet_score = []
    tfidf_cosine_score = []

    # find features
    dice_score.append(dice_recur(i,j))
    cosine_score.append(get_cosine(i,j))
    jaccard_score.append(is_ci_token_stopword_set_match(i,j))
    fuzzy_score.append(get_string_similarity(i,j))
    sequence_matcher.append(similar(i,j))
    # tf-idf cosine similarity
    tfidf_cosine_score.append(tfidf_cosine(i, j))
    rouge_score = calc_Rouge(i,j)
    surf_features = get_surface_features(i,j)
    fmeasure.append(FMeasure(j))
    #word2vec_scores.append(word2vec_score(i,j))
    ner_overlap_score = get_NER_overlap_score(i,j)
    num_overlap_score = get_number_similarity(i,j)
    # no of significant words + summation of significance values
    significant_score = getsignificance(j)
    # get best similarity between words based on wordnet
    wordnet_score.append(get_wordnet_based_similarity(i,j))
    

    features = tuple(dice_score) + tuple(cosine_score) + tuple(jaccard_score) + tuple(fuzzy_score) + tuple(sequence_matcher)+ tuple(tfidf_cosine_score) + rouge_score + tuple(fmeasure) + surf_features + tuple(wordnet_score) + ner_overlap_score + num_overlap_score + significant_score 
    featureVector = {}

    cnt = -1
    for feature in features:
        cnt +=1 
        featureVector[cnt] = feature

    return featureVector

        
featureSets = []
'''
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+".csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        feature = [(findFeature(x,y),labels) for x,y,labels in zip(a,b,label)]
        #tfidf = str(tfidf_cosine(a, b))
        featureSets = featureSets + feature
        
print(len(featureSets))
print(featureSets[1])
'''
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

        feature = [(findFeature(x,y),labels) for x,y,labels in zip(a,b,label)]
        #tfidf = str(tfidf_cosine(a, b))
        featureSets = featureSets + feature

# writing features to csv file
with open("test_no_Word2Vec_features.csv", "a", newline="") as f:
    cw = csv.writer(f)
    cw.writerows([v for _,v in sorted(d.items())] + [s] for d,s in featureSets)



    

