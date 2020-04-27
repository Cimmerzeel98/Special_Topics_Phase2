#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:23:48 2020

@author: florian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K

#print(tf.__version__)

def sinc(x):
    atzero = tf.ones_like(x)
    atother = tf.divide(tf.sin(x),x)
    value = tf.where(tf.equal(x,0), atzero, atother )
    return value

    
def build_model(train_dataset):
    
    
    model = keras.Sequential([
        layers.Dense(712, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(712, activation='relu'),
        layers.Dense(712, activation='relu'),
        layers.Dense(712, activation='relu'),
        layers.Dense(356, activation='tanh'),
        layers.Dense(356, activation='tanh'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='Nadam',
                metrics=['accuracy'])
    return model

#####################################################################################################


def error_alpha_prediction(Nmax,EPOCHS,Npts):    
    
    def norm(a):
        return  (a-np.min(a))/(np.max(a)-np.min(a))
    

    ######################## FIRST 50 TRAINING ELEMENTS SET UP ###########################
    
    alpha=norm([-70. , -67.5, -65. , -62.5, -60. , -57.5, -55. , -52.5, -50. ,
       -47.5, -45. , -42.5, -40. , -37.5, -35. , -32.5, -30. , -27.5,
       -25. , -22.5, -20. , -17.5, -15. , -12.5, -10. ,  -7.5,  -5. ,
        -2.5,   0. ])
        #,   2.5,   5. ,   7.5,  10. ,  12.5,  15. ,  17.5,
        #20. ,  22.5,  25. ,  27.5,  30. ,  32.5,  35. ,  37.5,  40. ,
        #42.5,  45. ,  47.5,  50. ,  52.5,  55. ,  57.5,  60. ,  62.5,
        #65. ,  67.5, 70.])
    error=norm([0.23281822, 0.18755359, 0.22322383, 0.28767419, 0.30056844,
       0.28616267, 0.23554024, 0.21157805, 0.21686775, 0.22908801,
       0.24512731, 0.25937801, 0.26557015, 0.22800007, 0.23038932,
       0.21432974, 0.18730447, 0.16167617, 0.14060268, 0.12717149,
       0.12271483, 0.12365485, 0.11902285, 0.10909013, 0.0984286 ,
       0.08891604, 0.08148453, 0.07673854, 0.01548944])
       #0.07673854,
       #0.08148453, 0.08891604, 0.0984286 , 0.10909013, 0.11902285,
       #0.12365485, 0.12271483, 0.12717149, 0.14060268, 0.16167617,
       #0.18730447, 0.21432974, 0.23038932, 0.22800007, 0.26557015,
       #0.25937801, 0.24512731, 0.22908801, 0.21686775, 0.21157805,
       #0.23554024, 0.28616267, 0.30056844, 0.28767419, 0.22322383,
       #0.18755359,0.23281822])
    
    
    ### Creation data set ###
    x = np.linspace(0,Npts-1,Npts)
    x = alpha
    y = error
    
    #print(data)
    
    ############################
    
    pdata = pd.DataFrame({'x':x[:],'y':y[:]})
    pdatapred= pd.DataFrame({'x':x[:],'y':y[:]})

    print('pdata',pdata)

    plt.plot(x,y)
    plt.show()
    
    ######################## BUILD AND TRAIN MODEL ###########################
   
    train_dataset = pdata.sample(frac=1, random_state=0)
    test_dataset = pdatapred.sample(frac=1, random_state=0)
    train_labels = train_dataset.pop('y')
    test_labels = test_dataset.pop('y')
    
    model = build_model(train_dataset)
    
    history = model.fit(train_dataset, train_labels, validation_split=0.2, epochs=EPOCHS)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    
    ######################## RESULTS ###########################
   
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss','val_loss'],loc='upper right')
    plt.show()
    test_predictions = model.predict(test_dataset).flatten()
 
    #a = plt.axes(aspect='equal')
    plt.close()
    plt.scatter(test_labels, test_predictions,s=7)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()
   
    print('test', test_dataset)
    print ('predictions',test_predictions)
    
    plt.close()
    plt.scatter(test_dataset, test_labels ,s=7, label='Fed values')
    plt.scatter(test_dataset,test_predictions, s=7, label='Model prediction')
    plt.legend(loc='best')
    plt.xlabel('Cone angle [-]')
    plt.ylabel('Euclidean error [-]')
    plt.show()


    
    
    
    

error_alpha_prediction(250,2000,250)
# for j in range(11,26):
#     sine_prediction(25,j)