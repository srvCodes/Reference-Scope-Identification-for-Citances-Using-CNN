import numpy as np 
import re, math
import csv 
import pandas as pd 
from collections import Counter

WORD = re.compile(r'\w+')

def get_tanimoto(vec1, vec2):
	intersection = set(vec1.keys()) & set(vec2.keys())
	#print(intersection)
	
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	only_vec1 = set(vec1.keys()) - intersection
	only_vec2 = set(vec2.keys()) - intersection

	#print(only_vec2)
	sum1 = sum([vec1[x] for x in only_vec1])
	sum2 = sum([vec2[x] for x in only_vec2])
	denominator = sum1 + sum2 + numerator

	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator
	
def text_to_vector(text):
	words = WORD.findall(text)
	return Counter(words)

def calc_tanimoto(file):
	dataframe = pd.read_csv(file, header = None, delimiter = "\t")
	a = dataframe[1]
	b = dataframe[2]
	
	score = []
	for i,j in zip(a,b):
		txt1 =  text_to_vector(i)
		txt2 =  text_to_vector(j)
		score.append(get_tanimoto(txt1, txt2))
		
	return score
'''
txt1 = "I am here now"
txt2 = "I will be there but not now"

vec1 = text_to_vector(txt1)
vec2 = text_to_vector(txt2)

print(get_tanimoto(vec1, vec2))
'''

