import re, math
from collections import Counter
import pandas as pd 
import csv

WORD = re.compile(r'\w+')

def text_to_vector(text):
     text = str(text)
     words = WORD.findall(text)
     return Counter(words)

def get_cosine(i, j):
     vec1 =  text_to_vector(i)
     vec2 =  text_to_vector(j)
     intersection = set(vec1.keys()) & set(vec2.keys())
     #print(intersection)
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


def calc_cosine(file):
	score = []
	for i,j in zip(a,b):
		
		score.append(get_cosine(txt1, txt2))
		
	return score
'''
txt1 = "I am here now."
txt2 = "I will be there tomorrow."

vec1 = text_to_vector(str(txt1))
vec2 = text_to_vector(str(txt2))

print(get_cosine(vec1, vec2))'''
