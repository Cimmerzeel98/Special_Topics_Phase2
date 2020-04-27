#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:57:55 2020

@author: Carmen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from Phase2w2_1 import ANN_init, ANN_target, error

def sinc(x):
    atzero = tf.ones_like(x)
    atother = tf.divide(tf.sin(x),x)
    value = tf.where(tf.equal(x,0), atzero, atother)
    return value

    
def build_model(train_dataset):
    
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='Nadam',
                metrics=['accuracy'])
    
    return model


#####################################################################################################


def error_prediction(Nmax,EPOCHS,Npts):    
    
    def norm(a):
        return  (a-np.min(a))/(np.max(a)-np.min(a))
    

    ######################## FIRST TRAINING ELEMENTS SET UP ###########################
    
    
    column_names = ['T initial', 'T target']
    
    #Obtain data from manifolds
    dataset= pd.DataFrame(data=np.transpose(np.vstack((norm(ANN_init),norm(ANN_target),norm(error)))),  columns = ['T initial', 'T final', 'Euclidean'])
    dataset.tail()


    ############################

    #Plot provided data
    plt.plot(ANN_init,error, label='t from initial')
    plt.scatter(ANN_target,error, s=0.1 , label='t to target')
    plt.legend(loc='best')
    plt.xlabel('Dimensionless TOF, [-]')
    plt.ylabel('Euclidean error, [-]')
    plt.grid()
    plt.show()
    
    ######################## BUILD AND TRAIN MODEL ###########################
    
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('Euclidean')
    test_labels = test_dataset.pop('Euclidean')
    #sns.pairplot(train_dataset[["T initial", "T final", "Euclidean"]], diag_kind="kde")
    
    model = build_model(train_dataset)
    
    history = model.fit(train_dataset, train_labels, validation_split=0.2, epochs=EPOCHS)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    
    #model.save('ST_model.h5')
    #model = load_model('ST_model.h5')
    
    ######################## RESULTS ###########################
    
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss','val_loss'],loc='upper right')
    plt.show()
    test_predictions = model.predict(test_dataset).flatten()
    
    plt.close()
    plt.scatter(test_labels, test_predictions,s=0.001)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()
   
    print('test', test_dataset)
    print ('predictions',test_predictions)
    print('minimum tof', min(test_predictions))
    print('L3 tof', )
    
    plt.close()
    plt.plot(norm(ANN_init),norm(error))
    plt.plot(test_dataset,test_predictions)
    plt.xlabel('tof')
    plt.ylabel('Euclidean error (normalised)')
    plt.show()
    

error_prediction(250,50,250)
# for j in range(11,26):
#     sine_prediction(25,j)