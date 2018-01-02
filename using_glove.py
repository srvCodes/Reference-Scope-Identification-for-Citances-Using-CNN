def loadGloveModel(gloveFile):
	print("Loading glove..")
	f = open(gloveFile, 'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = [float(val) for val in splitLine[1:]]
		model[word] = embedding
	print("Done", len(model), "words loaded!")
	f.close()
	return model

gloveFile = "glove.6B.100d.txt"
embeddings_index = loadGloveModel(gloveFile)
print(model['frog'])