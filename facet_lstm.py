import numpy as np 
import pandas as pd 
import re 
import nltk
import tensorflow as tf 
import random
import os 
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, SGD, Adadelta, Adamax
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

from collections import Counter
#from imblearn.over_sampling import SMOTE

print('loading data ..')
train = pd.read_csv('facet_train.csv', delimiter='\t', header=None)

print(train.shape)
train = train.dropna(axis=0)
X = train[0]
y = train[1]

#X = X.dropna(axis=0)
#y = y.dropna(axis=0)
y = y.astype('category')
y = y.cat.codes

#X = X.notnull()
#y = y.notnull()


encoder = LabelEncoder()
encoder.fit(y)
encoded_Ytrain = encoder.transform(y)
print(format(Counter(encoded_Ytrain))) 

print("preprocessing texts..")
print(len(X))

#print(X[:-1])
print(len(y))
def preprocessing_words(blog, remove_stopwords = False):
	# remove html tags
	blog_text = BeautifulSoup(blog, "html.parser").get_text()

	blog_text = re.sub("[^a-zA-Z0-9]", " ", blog_text)

	words = blog_text.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words =  [ w for w in words if not w in stopwords]

	return(" ".join(words))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def blogs_to_sentences(blog, tokenizer, remove_stopwords=False):
	# split a blog into parsed sentences. returns a list of sentnces
	# where each sentence is a list of words
	raw_sentences  = tokenizer.tokenize(blog.strip())

	# loop over each sentence
	#sentences = []
	#for raw_sentence in raw_sentences:
		# if a sentence is empty, skip it
	#	if(len(raw_sentence) > 0):
	#		# otherwise call preprocessing_words
	#		sentences.append(preprocessing_words(raw_sentence, remove_stopwords))
	sentences = [preprocessing_words(w, remove_stopwords) for w in raw_sentences if len(w) > 0]
 
	# return the list of sentences, ie. list of lists
	return(".".join(sentences))

sentences = []

print("Parsing sentences from train set.. ")

cnt = 0
for blog in X:
	X[cnt] = blogs_to_sentences(blog, tokenizer)
	cnt += 1

print(len(X))
#print(X)

train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X,y)

############# Down here: vocab building for lstm embeddings ############
'''
vocab = Counter()
for text in train_set_X:
	for word in nltk.word_tokenize(text):
		vocab[word.lower()] += 1

for text in test_set_X:
	for word in nltk.word_tokenize(text):
		vocab[word.lower()] += 1

total_words = len(vocab)
print("Total unique words", total_words)
texts = train_set_X + test_set_X
print("Total train samples: ", len(train_set_X))
print("Total test samples: ", len(test_set_X))


tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(texts.astype(str))
sequences = tokenizer.texts_to_sequences(texts.astype(str))
word_index = tokenizer.word_index

# For train data, converting to indexed-word-sequence
train_data_new = tokenizer.texts_to_sequences(train_set_X)
test_data_new = tokenizer.texts_to_sequences(test_set_X)
print('Found %s unique tokens.' % len(word_index))
#tokenizer = Tokenizer()'''

print("Indexing word vectors.. ")

EMBEDDING_DIM = 50
embeddings_index = {}
f = open(os.path.join('/home/saurav/Documents/nlp_intern','glove.6B.'+
	str(EMBEDDING_DIM)+'d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector
# ------------------------------------- #

# truncate and pad input text sequences
max_length = 300
X_train = sequence.pad_sequences(train_data_new, maxlen=max_length, padding='post')
X_test = sequence.pad_sequences(test_data_new, maxlen=max_length, padding='post')
y_train = train_set_y
y_test = test_set_y

print(X_train[1])

# Create the model
rmsprop = RMSprop(lr=0.01)
sgd = SGD(lr=0.1)
opt = Adamax(3.0)
num_epoch = 26
num_batch = 100

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
 	weights=[embedding_matrix], 
 	input_length=max_length,))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.7))
model.add(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.7))
model.add(LSTM(25))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model = load_model('checkpoints/model_gender_0.7365.h5')

print(model.summary())

print('Training model ...')
model.fit(X_train, y_train, epochs=num_epoch, batch_size=num_batch)
'''