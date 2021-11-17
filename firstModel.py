#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Note: what is the use of input and output activation function
"""
Created on Thu Aug 26 15:30:43 2021

@author: amit
"""

#Dependencies
import numpy as np
import pandas as pd
#dataset import
dataset = pd.read_csv('/home/amit/train.csv') #You need to change #directory accordingly
dataset.head(10) #Return 10 rows of data
#Changing pandas dataframe to numpy array
dataset_test= pd.read_csv('/home/amit/test.csv') #You need to change #directory accordingly
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values


#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# convert to binary classes
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# split the training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

#Dependencies

import keras
from keras.models import Sequential
from keras.layers import Dense


# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))


# Loss fucntion 
model.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy'])

#training model
#history = model.fit(X_train, y_train, epochs=100, batch_size=64)
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)


#testing the model
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

#testing accuracy of the model

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

# to check accuracy while training
#history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)

# visualize training losses and accuracies
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# visualize loss function

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()



Xt = dataset_test.iloc[:,1:21].values

Xt = sc.fit_transform(Xt)



test_pred=model.predict(Xt)
t_pred = list()
for i in range(len(test_pred)):
    t_pred.append(np.argmax(test_pred[i]))

#testing accuracy of the model
#a = accuracy_score(t_pred,t_test)
#print('Accuracy is:', a*100)

plt.plot(t_pred)
plt.show
#plt.plot(t_test)
#plt.show
