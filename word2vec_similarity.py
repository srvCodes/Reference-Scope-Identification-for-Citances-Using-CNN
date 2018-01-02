from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def word2vec_score(i,j):
	i = str(i)
	j = str(j)
	first = i.strip()
	first = first.split(" ")
	second = j.strip()
	second = second.split(" ")

	score = 0
	for i in first:
		try:
			for j in second:
				try:
					score += model.similarity(i,j)
				except KeyError:
					continue
		except KeyError:
			continue

	return score / (len(first) + len(second))

print(word2vec_score("I am a woman.", "The sun shines pinkish."))