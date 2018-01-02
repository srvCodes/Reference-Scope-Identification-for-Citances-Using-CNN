import nltk
import string
import re 
#from nltk.tag.stanford import StanfordPOSTagger

#stanford = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
#	'/home/saurav/Documents/postagger/stanford-postagger.jar')

def get_NER_tags(text):
	posTagList =  ['NN', 'NNP', 'NNS', 'NNPS']
	tokens = nltk.word_tokenize(text)
	#text = nltk.Text(tokens)
	tags = nltk.pos_tag(tokens)

	count = 0 

	for a,b in tags:
		if b in posTagList:
			count += 1

	return count

#text = "statistical methods, popular models Hidden Markov Models (HMM) (Rabiner, 1989), Maximum Entropy Models (ME) (Chieu et al., 2002) Conditional Random Fields (CRF) (Lafferty et al., 2001)."
#print(count)
