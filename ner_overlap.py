import nltk
import string
import re
from dice_coeff import dice_recur
from word2vec_similarity import word2vec_score
from fuzz_string_matching import get_string_similarity
from sequence_matcher import similar 
#from nltk.tag.stanford import StanfordPOSTagger

#stanford = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
#	'/home/saurav/Documents/postagger/stanford-postagger.jar')

def get_NER_overlap_score(citance, reference):
	posTagList =  ['NN', 'NNP', 'NNS', 'NNPS']
	tokens1 = nltk.word_tokenize(citance)
	tokens2 = nltk.word_tokenize(reference)
	#text = nltk.Text(tokens)
	tags_citance = nltk.pos_tag(tokens1)
	tags_reference = nltk.pos_tag(tokens2)

	count = 0 

	for a,b in tags_citance:
		if b in posTagList:
			count += 1
	ner_citances = [a for a,b in tags_citance if b in posTagList]
	ner_citances = " ".join(ner_citances)
	ner_references = [a for a,b in tags_reference if b in posTagList]
	ner_references = " ".join(ner_references)
	
	#print(ner_citances)
	#print(ner_references)

	dice_score = dice_recur(ner_citances, ner_references)
	word2vec_sim = word2vec_score(ner_citances, ner_references)
	fuzzy_score = get_string_similarity(ner_citances, ner_references)
	sequence_matcher_score = similar(ner_citances, ner_references)

	return (dice_score, word2vec_sim, fuzzy_score, sequence_matcher_score)

#a = "named entity recognizer (NER) useful many NLP applications information extraction, question answering, etc. own, NER also provide users looking person organization names quick information."
#b = "automatically derived based correlation metric value used (Chieu Ng, 2002a)."

#print(get_NER_overlap_score(a,b))


