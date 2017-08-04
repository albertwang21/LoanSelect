# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import keras

# Part 1
# Training & Testing Data Initialization
data = pd.read_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\cleaned_data.csv")
# len(data) = 403624
test = data[:80000] # about 20% used as testing
train = data[80000:] # will hold out another 80000 for validation
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
test.to_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\test_set.csv", index=False)
train.to_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\train_set.csv", index=False)

# Part 2
# Neural Network for Conditional Lifetime
train = pd.read_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\train_set.csv")
test = pd.read_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\test_set.csv")
train = train[train.loan_status == 1]
train = train.as_matrix()
test = test.as_matrix()
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
learning_rate = 0.001
Paramecium_lt = Sequential()
Paramecium_lt.add(Dense(800, init='he_normal', W_regularizer=l2(0.0001), bias=True, input_dim=369))
Paramecium_lt.add(Dense(2400, init='he_normal', W_regularizer=l2(0.0001), activation='tanh', bias=True))
Paramecium_lt.add(Dropout(0.5))
Paramecium_lt.add(Dense(1600, init='he_normal', W_regularizer=l2(0.0001), activation='tanh', bias=True))
Paramecium_lt.add(Dense(1, init='he_normal', W_regularizer=l2(0.0001), activation='sigmoid', bias=True))
sgd = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
Paramecium_lt.compile(optimizer=sgd, loss='binary_crossentropy')
model_checkpoint = ModelCheckpoint("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
hist1 = Paramecium_lt.fit(train[:, 2:371], train[:, 0], batch_size=64, nb_epoch=100, verbose=1, callbacks=[model_checkpoint, early_stopping], validation_split=0.2, shuffle=True)
np.save('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\hist1_loss.npy', hist1.history)
# hist1 = np.load('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\hist1_loss.npy').item()
sgd = SGD(lr=learning_rate/2, momentum=0.9, decay=1e-6)
Paramecium_lt.compile(optimizer=sgd, loss='binary_crossentropy')
hist2 = Paramecium_lt.fit(train[:, 2:371], train[:, 0], batch_size=64, nb_epoch=100, verbose=1, callbacks=[model_checkpoint, early_stopping], validation_split=0.2, shuffle=True)
np.save('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\hist2_loss.npy', hist2.history)
sgd = SGD(lr=learning_rate/4, momentum=0.9, decay=1e-6)
Paramecium_lt.compile(optimizer=sgd, loss='binary_crossentropy')
hist3 = Paramecium_lt.fit(train[:, 2:371], train[:, 0], batch_size=64, nb_epoch=100, verbose=1, callbacks=[model_checkpoint, early_stopping], validation_split=0.2, shuffle=True)
np.save('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\hist3_loss.npy', hist3.history)
# plot loss decreasing along epochs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.figure()
x = range(0,300)
y1 = loss1 + loss2 + loss3
y2 = val_loss1 + val_loss2 + val_loss3
plt.plot(x, y1, color='black', lw=2, label='training loss')
plt.plot(x, y2, '--', color='black', lw=2, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.show()



import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
plt.figure()
x = range(0,300)
y1 = hist1['loss'] + hist2.history['loss'] + hist3.history['loss']
y2 = hist1.history['val_loss'] + hist2.history['val_loss'] +hist3.history['val_loss']
plt.plot(x, y1, color='red', lw=2, label='loss')
plt.plot(x, y2, color='yellow', lw=2, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(loc="upper right")
plt.show()


# save prediction
conditional_lifetime_rt = Paramecium_lt.predict(test[:, 2:371], batch_size=64, verbose=1)
np.savetxt('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\lifetime_default.csv', conditional_lifetime_rt, fmt='%.20f')
Paramecium_lt.save_weights('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\Paramecium_lt_weights.h5')

# Part 3
# Neural Network for Default Risk
train = pd.read_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\train_set.csv")
test = pd.read_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\test_set.csv")
train = train.as_matrix()
test = test.as_matrix()
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
learning_rate = 0.005
Paramecium_dr = Sequential()
Paramecium_dr.add(Dense(800, init='he_normal', W_regularizer=l2(0.0001), bias=True, input_dim=369))
Paramecium_dr.add(Dense(2400, init='he_normal', W_regularizer=l2(0.0001), activation='tanh', bias=True))
Paramecium_dr.add(Dropout(0.5))
Paramecium_dr.add(Dense(1600, init='he_normal', W_regularizer=l2(0.0001), activation='tanh', bias=True))
Paramecium_dr.add(Dense(1, init='he_normal', W_regularizer=l2(0.0001), activation='sigmoid', bias=True))
sgd = SGD(lr=learning_rate, momentum=0.9, decay=1e-6)
Paramecium_dr.compile(optimizer=sgd, loss='binary_crossentropy')
model_checkpoint = ModelCheckpoint("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\weights.{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
cw = {0 : 1, 1: 4.3374}
hist1 = Paramecium_dr.fit(train[:, 2:371], train[:, 1], batch_size=128, nb_epoch=100, verbose=1, callbacks=[model_checkpoint, early_stopping], validation_split=0.2472, shuffle=True, class_weight=cw)
np.save('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\hist1_loss.npy', hist1.history)
# plot loss decreasing along epochs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
plt.figure()
x = range(0,61)
y1 = hist1.history['loss']
y2 = hist1.history['val_loss']
plt.plot(x, y1, color='red', lw=2, label='loss')
plt.plot(x, y2, color='yellow', lw=2, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(loc="upper right")
plt.show()
# load the best model
from keras.models import load_model
Paramecium_dr = load_model('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\weights.39-1.4748.hdf5')
# save prediction
Paramecium_dr.save_weights('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\Paramecium_dr_weights.h5')
default_risk = Paramecium_dr.predict_proba(test[:, 2:371], batch_size=128, verbose=1)
np.savetxt('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\default_risk.csv', default_risk, fmt='%.20f')




