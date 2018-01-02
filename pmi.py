# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:53:43 2017

@author: lenovo laptop
"""

from nltk import bigrams
import collections
import math

def getsignificance(a):
	a1=a.split() 
	a2=collections.Counter(a1)

	a3=collections.Counter(bigrams(a1))
	a4=sum([a2[x]for x in a2])
	a5=sum([a3[x]for x in a3])
	a6={x:float(a2[x])/a4 for x in a2} # word probabilities(w1 and w2)
	a7={x:float(a3[x])/a5 for x in a3} # joint probabilites (w1&w2)
	#u = {}
	
	v = collections.defaultdict(dict)
	for x in a6:
  		k={x:round(math.log((a7[b]/(a6[x] * a6[y])),2),4) for b in a7 for y in a6 if x in b and y in b}
  		v.update(k)

	res = dict((k, v) for k, v in v.items() if v >= 5.0)
	#print (sum(res.values()))
	#print(len(res))
	return (len(res), sum(res.values()))

a= """When the defendant and his lawyer walked into the court, some of the victim supporters turned their backs on him.  if we have more data then it will be more interesting because we have more chance to repeat bigrams. After some of the victim supporters turned their backs then a subset of the victim supporters turned around and left the court."""
print(getsignificance(a))