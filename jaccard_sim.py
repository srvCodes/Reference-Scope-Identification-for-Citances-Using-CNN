# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:14:47 2017

@author: lenovo laptop
"""
import nltk.corpus
from nltk.tokenize import TreebankWordTokenizer
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
import pandas as pd
import csv 

# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

# Create tokenizer and stemmer
#tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
tokenizer = TreebankWordTokenizer()

def is_ci_token_stopword_set_match(a, b):
    """Check if a and b are matches."""
    tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]

    # Calculate Jaccard similarity
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return (ratio)
'''
def main():
    a = "Global features are extracted from other occurrences of the same token in the whole document."
    b = "In such cases, neither global features (Chieu and Ng, 2002) nor aggregated contexts (Chieu and Ng, 2003) can help."
    print (is_ci_token_stopword_set_match(a, b))
  
if __name__ == "__main__":
  main()
'''
def calc_jaccard(file):
    dataframe = pd.read_csv(file, header = None, delimiter = "\t")
    a = dataframe[1]
    b = dataframe[2]
    score = []
    for i, j in zip(a,b):
        score.append(is_ci_token_stopword_set_match(i,j))

    return score