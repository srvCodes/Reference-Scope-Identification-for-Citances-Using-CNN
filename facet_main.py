import numpy as np 
import pandas as pd 
import re 
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

from collections import Counter
from imblearn.over_sampling import SMOTE

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

encoder = LabelEncoder()
encoder.fit(y)
encoded_Ytrain = encoder.transform(y)
print(format(Counter(encoded_Ytrain))) 


'''
max_features = 500
maxlen = 100
embedding_size = 128

kernel_size = 5
filters = 64
pool_size = 4

lstm_output_size = 70 

batch_size = 30
epochs = 2

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
''' '''
######### Manually padding by building vocabulary for characters ########
# Since keras.pad_sequences did not work

vocab_char_train = {k: (v+1) for k, v in zip(set(X_train), range(len(set(X_train))))}
vocab_char_train['<PAD>'] = 0 

rev_vocab_train = {v: k for k, v in vocab_char_train.items()}

vocab_char_test = {k: (v+1) for k, v in zip(set(X_test), range(len(set(X_test))))}
vocab_char_test['<PAD>'] = 0

rev_vocab_test = {v: k for k, v in vocab_char_test.items()}

x_train = [[vocab_char_train[char] for char in X_train[i:(i + maxlen)]] for i in range(0, len(X_train), maxlen)]
x_test = [[vocab_char_test[char] for char in X_test[i:(i + maxlen)]] for i in range(0, len(X_test), maxlen)]
print(len(x_train[1]))
print(len(x_test))

print("building model.. ")

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length = maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train =  np.reshape(x_train, (len(x_train), x_train.shape[1]))
x_test = np.reshape(x_test, (len(x_test, x_test.shape[1])))

print("training.. ")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test score: ", score)
print("Test accuracy: ", acc)
'''
'''
def preprocessing_words(blog, remove_stopwords = False):
	# remove html tags
	blog_text = BeautifulSoup(blog, "html.parser").get_text()

	blog_text = re.sub("[^a-zA-Z0-9]", " ", blog_text)

	words = blog_text.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words =  [ w for w in words if not w in stopwords]

	return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def blogs_to_sentences(blog, tokenizer, remove_stopwords=False):
	# split a blog into parsed sentences. returns a list of sentnces
	# where each sentence is a list of words
	raw_sentences  = tokenizer.tokenize(blog.strip())

	# loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
		# if a sentence is empty, skip it
		if(len(raw_sentence) > 0):
			# otherwise call preprocessing_words
			sentences.append(preprocessing_words(raw_sentence, remove_stopwords))

	# return the list of sentences, ie. list of lists
	return sentences

sentences = []

print("Parsing sentences from train set.. ")

for blog in X:
	sentences += blogs_to_sentences(blog, tokenizer)
'''