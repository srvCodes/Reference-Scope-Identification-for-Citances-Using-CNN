import numpy as np 
import pandas as pd
import math
import sklearn.metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LSTM, Embedding, Conv1D, Convolution2D, MaxPooling2D, GlobalAveragePooling1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import normalize, RobustScaler, LabelEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, f1_score
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

import elm

seed = 7
np.random.seed(seed)

dataframe = pd.read_csv("features.csv", header = None)
dataset = dataframe.values

#X_tr = dataset[:, 0:11].astype(float) 
X_tr = [x.astype(float) for i, x in enumerate(dataset[:, 0:]) if i < 11 and i > 25]
Y_tr = dataset[:,41]

test_set = pd.read_csv("test_features.csv", header = None)
dataset = test_set.values

X_test = dataset[:, 0:41].astype(float) 
y_test = dataset[:,41]

print(X_test)

'''
###### scale the features between 0 and 1 #######
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr)
X_test = scaler.fit_transform(X_test)

X_tr = normalize(X_tr, norm='l2')
X_test = normalize(X_test, norm='l2')

encoder = LabelEncoder()
encoder.fit(Y_tr)
encoded_Ytrain = encoder.transform(Y_tr)
print(format(Counter(encoded_Ytrain))) 

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_tr,Y_tr)
print(format(Counter(y_train)))

X_test, y_test = sm.fit_sample(X_test,y_test)
#print(format(Counter(y_test)))

pca = PCA(n_components=35)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
'''

############## 2D Convolutions  ==> working but bad ##################
'''  what we are trying to do is feed a vector of length 29 to CNN. X_train.shape[0] such vectors are there.
	We can imagine each of them as an image with height 1, width 29 and channels 1. Input 
	shape will be (1,29,1) as (rows,cols,channels). CNN1D input shape should 
	be (29,1). So reshaping X_Train as (339732 x 29 x 1) for CNN1D, 
	if using CNN2D then reshape as (339732 x 1 x 29 x 1).'''
'''
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
	
model = Sequential()

model.add(Convolution2D(32, (1, 1), input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(1))

start = 0.03
stop = 0.001
epochs = 20
learning_rate = np.linspace(start, stop, epochs)

sgd = SGD(lr=start, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(patience=100)

model.fit(X_train, y_train , batch_size=20, epochs=10, verbose = 1)
y_pred = model.predict(X_test)
print("########### precision here #########",precision_recall_fscore_support(y_test, y_pred.round()))
'''
'''
############################### LSTM network #######################################
X_tr = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
X_te = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(10, input_shape=(1, X_tr.shape[2])))  # returns a sequence of vectors of dimension 30
model.add(Dense(1))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(X_tr, y_train, batch_size = 400, epochs = 30, verbose = 1)
y_pred = model.predict(X_te)
print("########### precision here #########",precision_recall_fscore_support(y_test, y_pred.round()))
'''
'''
############# 1D Convolutional network ###############
print(type(X_train))
X_tr = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_te = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	
model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(X_tr.shape[1], 1)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_tr, y_train, validation_split = 0.1, batch_size=10, epochs=10, verbose = 1, callbacks=[EarlyStopping(patience=5,verbose=1), 
											ModelCheckpoint('checkpoint_best_epoch.hdf5',
										    save_best_only=True,
										  	verbose=1)])
									
y_pred = model.predict(X_te)
print("########### precision here #########",precision_recall_fscore_support(y_test, y_pred.round()))
'''
'''

############ simple dense network ###############
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

seed = 7
np.random.seed(seed)

model  = Sequential()
model.add(Dense(64, input_dim=30, kernel_initializer='normal', activation='relu'))
model.add(Dense(1))
#compiling the model
epochs = 20
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr = 0.0, momentum = 0.9, decay = 0.0, nesterov = False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
model.summary()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit(X_train, y_train, epochs = epochs, batch_size=30, callbacks = callbacks_list, verbose = 1)
y_pred = model.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred.round(), average='binary'))

grid_algo = GradientBoostingClassifier()
parameters = {'learning_rate':[0.06, 0.08, 0.1, 0.3, 0.5, 0.7]}

grid  = GridSearchCV(grid_algo, parameters, cv = 10, verbose = 0, scoring = 'f1')
grid.fit(X_train, y_train)

print("On all features:", grid.best_score_)
print(grid.best_params_)

alg = AdaBoostClassifier() #1.3

alg = alg.fit(X_train, y_train)
scores = cross_val_score(alg, X_test, y_test, cv = 10)
print("AdaBoost accuracy: ", sum(scores) / len(scores))

y_pred = alg.predict(X_test)
print(y_pred)
print(precision_recall_fscore_support(y_test, y_pred))

#alg1 = ExtraTreesClassifier()
#alg2 = RandomForestClassifier()
alg3 = GradientBoostingClassifier() # learning_rate=0.7
#alg4 = LinearSVC()

#eclf1 = VotingClassifier(estimators=[('ada', alg), ('gb', alg3)], voting = 'soft')
#eclf1 = eclf1.fit(X_train,y_train)

#y_pred = eclf1.predict(X_test)
#print(precision_recall_fscore_support(y_test, y_pred.round(), average='binary'))


alg1 = alg1.fit(X_train, y_train)
#scores = cross_val_score(alg1, X_test, y_test, cv = 10)
#print("Linear SVC accuracy: ", sum(scores) / len(scores))

y_pred = alg1.predict(X_test)
#print(y_pred)
print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#print("AUC_ROC score: %f" % auc(fpr, tpr))

alg2 = alg2.fit(X_train, y_train)
#scores = cross_val_score(alg2, X_test, y_test, cv = 10)
#print("Random Forest accuracy: ", sum(scores)/ len(scores))

y_pred = alg2.predict(X_test)
#print(sklearn.metrics.precision_recall_curve(y_test, y_pred))
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#print("AUC_ROC score: %f" % auc(fpr, tpr))
print(precision_recall_fscore_support(y_test, y_pred, average='micro'))

alg3 = alg3.fit(X_train, y_train)
#scores = cross_val_score(alg3, X_test, y_test, cv = 10)
#print("GradientBoosting accuracy: ", sum(scores)/len(scores))

y_pred = alg3.predict(X_test)
#print(y_pred)
print(precision_recall_fscore_support(y_test, y_pred))
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#print("AUC_ROC score: %f" % auc(fpr, tpr))


alg5 = LinearSVC()
alg5 = alg5.fit(X_train, y_train)
y_pred = alg5.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred))

alg6 = RandomForestClassifier()
alg6 = alg6.fit(X_train, y_train)
y_pred = alg6.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))


alg7 = MultinomialNB()
alg7 = alg7.fit(X_train, y_train)
y_pred = alg7.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred))

alg8 = LinearSVC(penalty='l1', dual=False)
alg8 = alg8.fit(X_train, y_train)
y_pred = alg8.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred))


'''