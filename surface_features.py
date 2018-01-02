import nltk
import string
import re 
import yuleK
import sentiWordNet
from ner_tagger import get_NER_tags

def get_surface_features(a,b):
	word_tokens_a = a.split(' ')
	word_tokens_b = b.split(' ')
	# count of words
	countWords_citances = len(word_tokens_a)
	countWords_references = len(word_tokens_b)
	# difference in count of words
	difference = abs(countWords_references - countWords_citances)
	# count of characters
	countCharacters_citances = len(a)-1
	countCharacters_references = len(b)-1
	# count of alphabets
	countAlphabets_citances = sum(c.isalpha() for c in a)
	countAlphabets_references = sum(c.isalpha() for c in b)
	albhabet_difference = abs(countAlphabets_references - countAlphabets_citances)
	# count of numbers
	countDigits_citances = sum(c.isdigit() for c in a)
	countDigits_references = sum(c.isdigit() for c in b)
	digits_difference = abs(countDigits_references - countDigits_citances)
	# count of spaces
	countSpaces_citances = sum(c.isspace() for c in a)
	countSpaces_references = sum(c.isspace() for c in b)
	# count of special characters
	countSpecialChars_citances = countCharacters_citances - countAlphabets_citances - countDigits_citances - countSpaces_citances
	countSpecialChars_references = countCharacters_references - countAlphabets_references - countDigits_references - countSpaces_references

	# long words are those whose lengths exceed 5 characters
	countLongWords_citances = sum(1 for word in word_tokens_a if len(word_tokens_a) > 5)
	countLongWords_references = sum(1 for word in word_tokens_b if len(word_tokens_b) > 5)
	difference_long_words = abs(countLongWords_citances - countLongWords_references)

	# count of punctuation marks
	countPunctuations_references = b.count('.') + b.count(',') + b.count('!') + b.count('?') + b.count(':') + b.count(';')
	count_doubleQuotes=re.findall(r'\"(.+?)\"', b)
	count_singleQuotes=re.findall(r'\'(.+?)\'', b)
	countPunctuations_references += len(count_singleQuotes) + len(count_doubleQuotes)
	normalizedPunctuations = countPunctuations_references * 1.0 / countCharacters_references

	lexicalRichness = yuleK.yule(b)

	# count of positive and negative words
	sentimentPOS_score, sentimentNEG_score = sentiWordNet.sentimentFeature(b)

	# average no of characters in each word 
	averageWordLength = sum(len(word) for word in word_tokens_b) / len(word_tokens_b)

	# no of named entities in the reference sentence
	num_of_NER_tags = get_NER_tags(b)

	return (countWords_references, difference, countCharacters_references, countAlphabets_references, countDigits_references, countSpaces_references, 
		countSpecialChars_references, countLongWords_references, difference_long_words, normalizedPunctuations, sentimentPOS_score, sentimentNEG_score, averageWordLength,
		lexicalRichness, num_of_NER_tags)

a = "statistical methods, popular models Hidden Markov Models (HMM) (Rabiner, 1989), Maximum Entropy Models (ME) (Chieu et al., 2002) Conditional Random Fields (CRF) (Lafferty et al., 2001)."
b = "Considerable amount work done recent years named entity recognition task, partly due Message Understanding Conferences (MUC)."

print(get_surface_features(a,b))
