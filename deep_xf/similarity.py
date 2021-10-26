#! /usr/bin/env python
"""
@author: Ajay Arunachalam
Created on: 13/10/2021
Goal: Filtered and Original Raw Time-series data Similarity with Siamese Neural Network using keras
Version: 0.0.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Subtract, Multiply, Concatenate, Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
import warnings
warnings.filterwarnings('ignore')

class Siamese:

	def Siamese_Model(features):

	    """
	    Implementation of the Siamese Network
	    Args:
	        features (int): number of features 
	    Returns:
	        [keras model]: siamese model
	    """

	    inp1 = Input(shape=(features,))
	    inp2 = Input(shape=(features,))

	    diff = Subtract()([inp1,inp2])
	    # squared difference
	    L2 = Multiply()([diff,diff])                     
	    # product proximity
	    prod = Multiply()([inp1,inp2])                  
	    # combined metric
	    combine = Concatenate(axis=1)([L2,prod])                

	    path1 = Dense(64)(L2)
	    path1 = BatchNormalization()(path1)
	    path1 = Dropout(0.25)(path1)
	    path1 = Activation('relu')(path1)

	    path2 = Dense(64)(prod)
	    path2 = BatchNormalization()(path2)
	    path2 = Dropout(0.25)(path2)
	    path2 = Activation('relu')(path2)

	    path3 = Dense(64)(combine)
	    path3 = BatchNormalization()(path3)
	    path3 = Dropout(0.25)(path3)
	    path3 = Activation('relu')(path3)

	    # combining everything
	    paths = Concatenate(axis=1)([path1,path2,path3])        


	    top = Dense(256)(paths)
	    top = BatchNormalization()(top)
	    top = Dropout(0.25)(top)
	    top = Activation('relu')(top)


	    out = Dense(1)(top)   # output similarity score                          

	    siamese_model = Model(inputs=[inp1,inp2],outputs=[out])

	    return siamese_model