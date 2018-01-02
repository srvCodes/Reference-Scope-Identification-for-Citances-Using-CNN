from nltk.corpus import wordnet 
from itertools import product 

#  check the similarity between each words in the two list and find out the maximum similarity
def get_wordnet_based_similarity(i, j):
	i = i.strip()
	j = j.strip()
	list1 = i.split(" ")
	list2 = j.split(" ")
	#print(list1)
	#print(list2)
	'''
	for word1 in list1:
		for word2 in list2:
			wordFromList1 = wordnet.synsets()'''

	allsyns1 = set(ss for word in list1 for ss in wordnet.synsets(word))
	allsyns2 = set(ss for word in list2 for ss in wordnet.synsets(word))
	try:
		best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in 
        	product(allsyns1, allsyns2))
		return best[0]
	except:
		return 0 


a = "named entity recognizer (NER) useful many NLP applications information extraction, question answering, etc. own, NER also provide users looking person organization names quick information."
b = "automatically derived based correlation metric value used (Chieu Ng, 2002a)."

print(get_wordnet_based_similarity(a,b))
