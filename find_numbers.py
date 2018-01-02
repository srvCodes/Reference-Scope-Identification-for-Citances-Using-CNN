# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:35:14 2017

@author: lenovo laptop
"""
import re 
from dice_coeff import dice_recur
from cosine import get_cosine
from jaccard_sim import is_ci_token_stopword_set_match
from fuzz_string_matching import get_string_similarity
from rouge_score import calc_Rouge
from sequence_matcher import similar 

def find_numbers(s):
    numbers = []
    
    numbers = re.sub("[^0-9]", " ", s)

    words = numbers.split()
    words = " ".join(words)
 
    return words

#a = "cases, neither global features (Chieu Ng, 2002) aggregated contexts (Chieu Ng, 2003) help."
#b = "local features used similar used BBN' IdentiFinder (Bikel et al., 1999) MENE (Borthwick, 1999)."


def get_number_similarity(citances, references):
	numbers_citances = find_numbers(citances)
	numbers_references = find_numbers(references)
	fuzzy_score = get_string_similarity(numbers_citances, numbers_references)
	sequence_matcher_score = similar(numbers_citances, numbers_references)
	return(fuzzy_score, sequence_matcher_score)

#print(get_number_similarity(a,b))