from __future__ import division
import nltk
from collections import Counter


from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
	'/home/saurav/Documents/postagger/stanford-postagger.jar')

def FMeasure(text):
	#text = "Hey bro! Can I get your cool shoes?"
	#text = text.decode('utf-8')
	totalWords = len(text.split())

	tokens = nltk.word_tokenize(text)
	text = nltk.Text(tokens)
	#tags = st.tag(text)
	tags = nltk.pos_tag(tokens)

	posCounts = Counter(tag for word,tag in tags)

	countNoun = posCounts['NN']+posCounts['NNS']+posCounts['NNP']+posCounts['NNPS']+posCounts['FW']
	countAdj = posCounts['CD']+posCounts['JJ']+posCounts['JJR']+posCounts['JJS']
	countPrep = posCounts['IN']+posCounts['TO']
	countArt = posCounts['DT']
	countPron = posCounts['EX']+posCounts['PRP']+posCounts['PRP$']+posCounts['WDT']+posCounts['WP']+posCounts['WP$']
	countVerb = posCounts['MD']+posCounts['VB']+posCounts['VBZ']+posCounts['VBP']+posCounts['VBD']+posCounts['VBN']+posCounts['VBG']
	countAdverb = posCounts['RB']+posCounts['RBR']+posCounts['RBS']+posCounts['RP']+posCounts['WRB']
	countIntj = posCounts['UH']

	freqNoun = countNoun*100.0/totalWords
	freqAdj = countAdj*100.0/totalWords
	freqPrep = countPrep*100.0/totalWords
	freqArt = countArt*100.0/totalWords
	freqPron = countPron*100.0/totalWords
	freqVerb = countVerb*100.0/totalWords
	freqAdverb = countAdverb*100.0/totalWords
	freqIntj = countIntj*100.0/totalWords

	FMeasure = (freqNoun+freqAdj+freqPrep+freqArt-freqPron-freqVerb-freqAdverb-freqIntj+100)/2

	return FMeasure*1.0/100

print(FMeasure("statistical methods, popular models Hidden Markov Models (HMM) (Rabiner, 1989), Maximum Entropy Models (ME) (Chieu et al., 2002) Conditional Random Fields (CRF) (Lafferty et al., 2001)."))