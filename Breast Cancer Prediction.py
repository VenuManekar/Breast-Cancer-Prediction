from calendar import EPOCH
from matplotlib import units
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv('Data.csv')

def f(i):
    if i == 2:
        return 0 
    elif i == 4:
        return 1

df.Class = df.Class.apply(f)
print(df.head())

from sklearn.model_selection import train_test_split
train_X, test_X,train_y, test_y = train_test_split(df.iloc[:, 1:-1], df.iloc[ :, -1],test_size= 0.2, random_state= 0)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units = 9, activation = 'relu')) 

model.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))
 
model.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_X,train_y, batch_size = 32,epochs = 25)
print('evaluate')

model.evaluate(test_X, test_y)
