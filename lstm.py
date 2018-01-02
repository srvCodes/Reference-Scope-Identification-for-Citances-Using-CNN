import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.layers import LSTM, Activation, Dense, Merge
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop 
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding 
from keras.backend import tf 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support 
from collections import Counter
from imblearn.over_sampling import SMOTE

seed = 7
np.random.seed(seed)

dataframe = pd.read_csv("features.csv", header = None)
dataset = dataframe.values

X = dataset[:, 0:13].astype(float)
Y = dataset[:,13]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(format(Counter(encoded_Y))) 

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X,encoded_Y)
print(format(Counter(y_res)))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state = 123)

#baseline model
def create_baseline():
	#create model
	model  = Sequential()
	model.add(LSTM(14, input_shape=(13,1), return_sequences = True))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#compile the model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model 

#now evaluate above model using stratified k fold cv with standardized dataset
estimator = KerasClassifier(build_fn = create_baseline, nb_epoch = 100, batch_size=5, verbose = 0)
#bdt = AdaBoostClassifier(base_estimator = estimator, algorithm = 'SAMME')
#bdt.fit(X_train, y_train)
#print(bdt.score(X_test, y_test))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv = kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
