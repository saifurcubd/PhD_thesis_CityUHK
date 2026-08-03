import argparse
import os
import time
import torch
import torch.nn as nn
import pickle
import logging


import numpy as np
import pandas as pd
import csv
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc,roc_auc_score,precision_recall_curve,mean_squared_error
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import pickle

import matplotlib.pyplot as plt
import torch
import json
import random

import warnings
warnings.filterwarnings("ignore")


from datetime import datetime
from torchsummary import summary

import numpy as np
from sklearn.preprocessing import normalize
def to01(array):
    a = array.min()
    # ignore the Runtime Warning
    with np.errstate(divide='ignore'):
        b = 1. /(array.max() - array.min())
    if not(np.isfinite(b)):
        b = 0
    return np.vectorize(lambda x: b * (x - a))(array)

        
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # (3, 64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (64, 16, 16)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # Adjust based on image dimensions
        self.fc2 = nn.Linear(128, 1)  # Binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
        
        
class DrugSynergyCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(DrugSynergyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding='same')
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding='same')
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2432, 1024)  # Adjust size based on final feature map size
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x):
        x=to01(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout(x)
        #x = nn.ReLU()(self.fc3(x))
        x = nn.ReLU()(self.bn1(self.fc3(x)))
        x = self.fc4(x)
        return x
        
        
class Model_ANN(nn.Module):
    def __init__(self, input_size):
        super(Model_ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        print("MACSynDCR_ANN model:")
        #summary(model, , batch_size=-1, device='cuda')
        #summary(model, , batch_size=-1, device='cuda')
        #summary(self, input_size=(128, 2176))
        #print(summary(self,128,2176))
        return x
        
        
class MACSynDCR_GRU2(nn.Module):
    def __init__(self):
        super(MACSynDCR_GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=2176, hidden_size=32, batch_first=True, dropout=0.2)
        self.gru2 = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1) 
        self.relu = nn.ReLU()        # For binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
       # x = self.flatten(x[:, -1, :])  # Use the last time step
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        x = self.sigmoid(self.fc3(x))
        return x
        

        

class MACSynDCR_CNN(nn.Module):
    def __init__(self,batch_size):
        super(MACSynDCR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(batch_size*68*32, 64)  #total input size according to batch (32,68,32)
        # self.fc1 = nn.Linear(32 * 17 * 8, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
        
class MACSynDCR_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MACSynDCR_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.unsqueeze(1)  # Adding sequence length dimension
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return self.sigmoid(out)
                
        
        
def LSTM_model2():
    model = Sequential([
        Input(shape=(68,32)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        LSTM(64),
        BatchNormalization(),
        Dense(32, activation='sigmoid'),
        BatchNormalization(),
        Dense(1, activation='softmax')
    ])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    print("MACSynDCR_LSTM model:")
    model.summary()
    return model


def DNN_model(input_size):
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])
    model.summary()
    return model
    
    
    
def CNN_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(68, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])
    return model
    
def GRU_model():
    # Input shape should match your data (timesteps, features)
    inputs = Input(shape=(68, 32))  # Define the input layer
    x = LSTM(units=128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(units=64)(x)
    x = BatchNormalization()(x)
    outputs = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)  # Create the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    model.summary()
    return model
    

