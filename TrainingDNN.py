"""
---------------------------------------------------------
Author: Brahim mefgouda
Email: brahim.mefgouda@ieee.org 
Affiliation: 6G Research Center, Khalifa University, UAE 
Date: October 2024
---------------------------------------------------------
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import std
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Lambda, Input, Dense, Dropout, BatchNormalization
import numpy as np

def average_flipped_bits_and_accuracy(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Both arrays must have the same shape.")
    
    flipped_bits = (X != Y).sum(axis=1)
    threshold = 3  # 10% of 16
    
    total_flipped_bits = flipped_bits.sum()
    num_pairs = X.shape[0]
    average_flipped = total_flipped_bits / num_pairs
    
    accuracy = (flipped_bits < threshold).mean() * 100
    total_flipped_bits_ten_perc_cent = (flipped_bits >= threshold).mean()
    
    return average_flipped, total_flipped_bits_ten_perc_cent, accuracy


#%% Load data sss

def Training_DNN():        
    LinkOfDatasetPUF= 'CRP_FPGA_01.csv'
    dataPUF=pd.read_csv(LinkOfDatasetPUF,encoding= 'utf-8')
    LinkOfDatasetNumbers= 'GeneratorDataset.csv'
    dataNumbers=pd.read_csv(LinkOfDatasetNumbers,encoding= 'utf-8')
    n_inputs   = 36 
    n_outputs  = 48
    print(dataNumbers)
    
    X_size = Input(shape=(n_inputs,))
    Model1Model = Dense(32, activation='relu', name='L6')(X_size )
    Model1Model = Dense(64, activation='relu', name='L62')(Model1Model )
    Model1Model = Dense(128, activation='relu', name='L6Z')(Model1Model )
    Model1Model = Dense(256, activation='relu', name='L7E')(Model1Model )
    Model1Model = Dense(256*2, activation='relu', name='L8')(Model1Model )
    Model1Model = Dense(1024, activation='relu', name='L9')(Model1Model )
    Model1Model = Dense(n_outputs, activation='linear')(Model1Model ) # relu non , sigmoid ? 
    
    Model1Build = Model(X_size, Model1Model, name='Model')
    
    # Define the autoencoder model
    Model1Build = Model(inputs=X_size, outputs= Model1Model )
    
    # Compile the model with an appropriate optimizer and loss function
    Model1Build.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    
    history = Model1Build.fit(dataNumbers,dataPUF, epochs=500, batch_size=100,  validation_data=(dataNumbers,dataPUF) )
    Model1Build.save('puf_model.h5')

def GenerateChallangeResponse(DataSetofIndex='GeneratorDataset.csv' , model='DNN.h5'):
    model = load_model( model)
    
    #DatasetOfIndex= 'GeneratorDataset.csv'
    DataSetofIndex=pd.read_csv(DataSetofIndex,encoding= 'utf-8')
    #LinkOfDatasetPUF= 'CRP_FPGA_01.csv'
    DataSetofIndex= np.array(DataSetofIndex)
    #DataSetofIndex= DataSetofIndex.reshape(1, -1)
    prediction = model.predict(DataSetofIndex)
    return prediction.round() 


def TestTheGeneratio( GeneratedData,OriginalData): 
    return (average_flipped_bits_and_accuracy(data,data ))
    

def GetDnn(): 
    model='DNN.h5'
    return load_model( model)

Training_DNN()
data = GenerateChallangeResponse()
LinkOfDatasetPUF= 'CRP_FPGA_01.csv'
OriginalData=pd.read_csv(LinkOfDatasetPUF,encoding= 'utf-8')

print (f'Accurcy: {TestTheGeneratio(data,OriginalData )[2]}')
