# It is always good to run in jupyter notebook for better data visualization.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, LeakyReLU, Input, Dropout, Dense, Add, Dropout
from tensorflow.keras import Model, datasets, models
from tensorflow.keras.optimizers import Adam

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('creditcard.csv')

data.tail()

data.shape

data.isnull().sum()

data.info()

data['Class'].value_counts()

# This is Highly inbalance data as for non-fraud=284315 and for fraud=492 so we need to balance it

non_fraud = data[data['Class']==0]
fraud = data[data['Class']==1]

non_fraud.shape, fraud.shape

# Now we will randomly select 492 samples from non_fraud such that samples in fraud and non_fraud will be same

non_fraud = non_fraud.sample(fraud.shape[0])

# Now it is balanced dataset
non_fraud.shape, fraud.shape

# Merging fraud and non_fraud data
new_data = fraud.append(non_fraud, ignore_index=True)

new_data['Class'].value_counts()

new_data

# saperating features and predicting value
# x contains our all featurs
# y contains output which needs to be predicted
# In Class or y, 0=non_fraud and 1=fraud

x = new_data.drop('Class', axis=1)
y = new_data['Class']

# spliting data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

# Feature scaling with mean normalization for x data
# for x_test only transform is used to avoid overfitting

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# converting y data into numpy

y_train = np.array(y_train)
y_test = np.array(y_test)

# reshape data as keras model takes 3-D data i.e. expanding 1 dimension

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# CNN MODEL

# Convolutional Neural Network

init = tf.random_normal_initializer(0.,0.2)

def fraud():
    I = Input(shape=x_train[0].shape)
    
    C1 = Conv1D(32, 2, kernel_initializer=init)(I)
    B1 = BatchNormalization()(C1)
    L1 = LeakyReLU()(C1)
    D1 = Dropout(0.5)(L1)
    
    C2 = Conv1D(64, 2, kernel_initializer=init)(L1)
    B2 = BatchNormalization()(C2)
    L2 = LeakyReLU()(B2)
    D2 = Dropout(0.5)(L2)
    
    F3 = Flatten()(D2)   
    DE3 = Dense(64)(F3)
    L3 = LeakyReLU()(DE3)
    D3 = Dropout(0.5)(L3)

    
    out = Dense(1, activation='sigmoid')(D3)
    
    model = Model(inputs=I, outputs=out)
    
    return model
    

model = fraud()
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train = model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=30)

# Plots to display loss and accuracy

plt.figure()
plt.plot(train.history['accuracy'])
plt.plot(train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred = model.predict(x_test)

# Prediction
np.round(pred.astype('int32'))
