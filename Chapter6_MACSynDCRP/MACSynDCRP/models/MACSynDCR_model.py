import numpy as np
import pandas as pd
import csv
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import keras
from keras import backend as K

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, LSTM, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras.layers import BatchNormalization
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ELU, PReLU, LeakyReLU
from tensorflow.keras.optimizers import RMSprop,Adam, SGD
import tensorflow as tf
from sklearn.utils import class_weight
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
from keras.models import load_model

import argparse
import os
import time
import pickle
import logging


import numpy as np
import pandas as pd
import csv
import glob
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc,roc_auc_score,precision_recall_curve,mean_squared_error
import numpy as np
import pandas as pd
import json
import random
from torchsummary import summary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    auc,
    precision_recall_curve,
    mean_squared_error
)
loss_func = nn.MSELoss(reduction='sum')

import warnings
warnings.filterwarnings("ignore")


from datetime import datetime

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

    
    
# Py Torch ANN     
class MACSynDCR_ANN(nn.Module):
    def __init__(self):
        super(MACSynDCR_ANN, self).__init__()
        self.fc1 = nn.Linear(2432, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        #self.fc2 = nn.Linear(2048, 1024)
        #self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.fc1(x)))
        #x = nn.ReLU()(self.bn2(self.fc2(x)))
        x = nn.ReLU()(self.bn3(self.fc3(x)))
        x = nn.ReLU()(self.bn4(self.fc4(x)))
        x = nn.ReLU()(self.bn5(self.fc5(x)))
        x = F.softmax(self.fc6(x), dim=1)  # Softmax for output layer
        return x
       
        
class MACSynDCR_CNN(nn.Module):
    def __init__(self):
        super(MACSynDCR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(32 * 68 * 32, 128)  # Adjusted for output shape of conv layers
        self.fc2 = nn.Linear(128, 1)  # Binary output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
       
        
#Keras ANN
def MACSynDCR_ANN_model2():
    model = keras.Sequential([
        layers.Input(shape=(2432,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    print("MACSynDCR_ANN model:")
    model.summary()
    return model

def MACSynDCR_ANN_model():
    model = keras.Sequential([
        layers.Input(shape=(2432,)),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    print("MACSynDCR_ANN model:")
    model.summary()
    return model

#Keras CNN
def MACSynDCR_CNN_model():
    model1 = Sequential()
    model1.add(Conv2D(32, (3, 3), input_shape = (64,38,1), activation='relu'))
    model1.add(MaxPooling2D(pool_size = (2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Conv2D(64, (3, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size = (2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Conv2D(128, (3, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size = (2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(128, activation = 'relu'))
    model1.add(Dense(2, activation = 'softmax'))
    
    print("MACSynDCR_CNN model:")
    model1.summary()
    return model1
   
        


    
    
def Ensemble_MACSynDCR_model2(ann_preds, cnn_preds,y_test): 
    #ensemble_pred=np.array(preds).T
    print(ensemble_pred)
    summed = np.sum(ensemble_pred, axis=0)
    # argmax across classes
    mean_pred = np.mean(ensemble_pred, axis=1)
    #ensemble=[]
    j=0
    #for i in 1:
     #   ensemble.append(ensemble_pred[j,i])
     #   j=j+1
    ensemble=mean_pred
    print(ensemble)
    ensemble = (ensemble > 0.5).astype(int)
    ensemble_pred = (ensemble_pred > 0.5).astype(int)
    accuracy1 = accuracy_score(y_test, ensemble_pred[:,0])
    accuracy2 = accuracy_score(y_test, ensemble_pred[:,1])
    #accuracy3 = accuracy_score(y_test, ensemble_pred[:,2])
    #accuracy4 = accuracy_score(y_test, ensemble_pred[:,3])
    #accuracy5 = accuracy_score(y_test, ensemble_pred[:,4])
    ensemble_accuracy = accuracy_score(y_test, ensemble)

    print('Accuracy Score for Ensemble = ', accuracy1)
    print('Accuracy Score for LSTM = ', accuracy2)
    #print('Accuracy Score for LSTM = ', accuracy3)
    #print('Accuracy Score for LSTM = ', accuracy4)
    #print('Accuracy Score for GRU = ', accuracy5)
    print('Accuracy Score for ensemble = ', ensemble_accuracy)
    
    #Weighted average ensemble
    weights = [0.5, 0.3, 0.2]

    #Use tensordot to sum the products of all elements over specified axes.
    weighted_preds = np.tensordot(ensemble_pred, weights, axes=((0),(0)))
    weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)

    weighted_accuracy = accuracy_score(y_test, weighted_ensemble_prediction)

    print('Accuracy Score for model1 = ', accuracy1)
    print('Accuracy Score for model2 = ', accuracy2)
    #print('Accuracy Score for model3 = ', accuracy3)
    print('Accuracy Score for average ensemble = ', ensemble_accuracy)
    print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)

    import pandas as pd
    df = pd.DataFrame([])

    for w1 in range(0, 5):
      #  for w2 in range(0,5):
            for w2 in range(0,5):
                wts = [w1/10.,w2/10.]
                wted_preds1 = np.tensordot(ensemble_pred, wts, axes=((0),(0)))
                wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
                weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
                df = pd.concat([df,pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                             'acc':weighted_accuracy*100}, index=[0])], ignore_index=True)

                #df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
       
    max_acc_row = df.iloc[df['acc'].idxmax()]
    print("Max accuracy of ", max_acc_row[0], " obained with w1=", max_acc_row[1],
          " w2=", max_acc_row[2])  

    max(df['acc'])
    
    return ensemble
    
def Ensemble_MACSynDCR_model(train_pred,ann_preds,kann_preds, kcnn_preds,train_labels,y_test, fold): 
    # Prepare features for stacking
    tann_preds=train_pred[0]
    tkann_preds=train_pred[1][:,1]
    tkcnn_preds=train_pred[2][:,1]
    
    ann_auc = roc_auc_score(y_test, ann_preds)
    ann_accuracy = accuracy_score(y_test, (ann_preds > 0.5).astype(int))
    print(f'Accuracy Score and AUC for ANN = AUC: {ann_auc:.4f}, Accuracy: {ann_accuracy:.4f}')
    
    kann_auc = roc_auc_score(y_test, kann_preds)
    kann_accuracy = accuracy_score(y_test, (kann_preds > 0.5).astype(int))
    print(f'Accuracy Score and AUC for KANN = AUC: {kann_auc:.4f}, Accuracy: {kann_accuracy:.4f}')
    
    kcnn_auc = roc_auc_score(y_test, kcnn_preds)
    kcnn_accuracy = accuracy_score(y_test, (kcnn_preds> 0.5).astype(int))
    print(f'Accuracy Score and AUC for KCNN = AUC: {kcnn_auc:.4f}, Accuracy: {kcnn_accuracy:.4f}')
    
    
    #stacked_X = np.column_stack((ann_preds,kann_preds, kcnn_preds))
    train_X = np.column_stack((tann_preds,tkann_preds, tkcnn_preds))
    test_X = np.column_stack((ann_preds,kann_preds,kann_preds))
    print(train_X.shape, type(test_X.shape))
    
    avg_pred=np.mean(test_X, axis=1) 
    #mean_pred = np.mean(ensemble_pred, axis=1)
    max_pred=np.max(test_X, axis=1)     
    avg_auc = roc_auc_score(y_test, avg_pred)
    avg_accuracy = accuracy_score(y_test, (avg_pred > 0.5).astype(int))
    print(f'Average AUC = AUC: {avg_auc:.4f}, Accuracy: {avg_accuracy:.4f}')
    
    max_auc = roc_auc_score(y_test, max_pred)
    max_accuracy = accuracy_score(y_test, (max_pred > 0.5).astype(int))
    print(f'MAX AUC = AUC: {max_auc:.4f}, Accuracy: {max_accuracy:.4f}')
    
    
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
       'n_estimators': [50, 200],
       'max_depth': [None, 10, 30],
       'min_samples_split': [2, 10],
       'min_samples_leaf': [1, 4],
    }
    
    #grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                              # scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
    #grid_search.fit(train_X, train_labels)
   # best_rf_model = grid_search.best_estimator_
    #rf_ensemble_prob = best_rf_model.predict_proba(test_X)[:, 1]
    
    
    #{'C': [0.01, 0.1, 1, 10, 100]
    param_grid = {'C': [0.01,0.1, 1, 10,100], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(train_X, train_labels)
    best_model = grid_search.best_estimator_
    #meta_model = LogisticRegression()
    #meta_model.fit(stacked_X, y_test)
    lr_ensemble_prob = best_model.predict_proba(test_X)[:,1]
    
    #rf_accuracy = accuracy_score(y_test, (rf_ensemble_prob> 0.5).astype(int))
    lr_accuracy = accuracy_score(y_test, (lr_ensemble_prob> 0.5).astype(int))
    if 0>=lr_accuracy:
        ensemble_prob=rf_ensemble_prob
        ensemble_pred=(ensemble_prob > 0.5).astype(int)
    else:
        ensemble_prob=lr_ensemble_prob
        ensemble_pred=(ensemble_prob > 0.5).astype(int)
    
    print(f"Finall ensemble results of fold_{fold+1} for MACSynDCR:")
    logging.info(f"Finall ensemble results of fold_{fold+1} for MACSynDCR:")
    logging.info("-" * 100)
    print("-" * 100)
    auc = roc_auc_score(y_test, ensemble_prob)
    accuracy = accuracy_score(y_test, ensemble_pred)
    auc_pr = average_precision_score(y_test,ensemble_prob)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    yp=torch.tensor(ensemble_pred.astype(float))
    rmse = np.sqrt(loss_func(y_test.float(), yp))
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc, _ = pearsonr(y_test, ensemble_prob)
            
    print(f'Ensemble Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nEnsemble AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nEnsemble Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Ensemble RMSE: {rmse:.4f} , PCC: {pcc:.4f}')
    logging.info(f'Ensemble Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}, \nEnsemble AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nEnsemble Recall: {recall:.4f}, F1 Score: {f1:.4f}, \nEnsemble RMSE: {rmse:.4f}')
            
    logging.info("-" * 100)
    print("-" * 100)
    return ensemble_pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc
    
    
def Ensemble_MACSynDCR_model3(train_pred,ann_preds,kann_preds, kcnn_preds,train_labels,y_test, fold): 
    # Prepare features for stacking
    tann_preds=train_pred[0]
    tkann_preds=train_pred[1][:,1]
    tkcnn_preds=train_pred[2][:,1]
    ann_auc = roc_auc_score(y_test, ann_preds)
    ann_accuracy = accuracy_score(y_test, (ann_preds.numpy() > 0.5).astype(int))
    print(f'Accuracy Score and AUC for ANN = AUC: {ann_auc:.4f}, Accuracy: {ann_accuracy:.4f}')
    
    kann_auc = roc_auc_score(y_test, kann_preds)
    kann_accuracy = accuracy_score(y_test, (kann_preds > 0.5).astype(int))
    print(f'Accuracy Score and AUC for KANN = AUC: {kann_auc:.4f}, Accuracy: {kann_accuracy:.4f}')
    
    kcnn_auc = roc_auc_score(y_test, kcnn_preds)
    kcnn_accuracy = accuracy_score(y_test, (kcnn_preds> 0.5).astype(int))
    print(f'Accuracy Score and AUC for KCNN = AUC: {kcnn_auc:.4f}, Accuracy: {kcnn_accuracy:.4f}')
    
    
    #stacked_X = np.column_stack((ann_preds,kann_preds, kcnn_preds))
    train_X = np.column_stack((tann_preds,tkann_preds, tkcnn_preds))
    test_X = np.column_stack((ann_preds,kann_preds, kcnn_preds))
    
    print(train_X.shape, test_X.shape)
    
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
       'n_estimators': [50, 200],
       'max_depth': [None, 10, 30],
       'min_samples_split': [2, 10],
       'min_samples_leaf': [1, 4],
    }
    
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                               scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
    grid_search.fit(train_X, train_labels)
    best_rf_model = grid_search.best_estimator_
    rf_ensemble_prob = best_rf_model.predict_proba(test_X)[:, 1]
    
    
    #{'C': [0.01, 0.1, 1, 10, 100]
    param_grid = {'C': [0.01, 1, 100], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3)
    grid_search.fit(train_X, train_labels)
    best_model = grid_search.best_estimator_
    #meta_model = LogisticRegression()
    #meta_model.fit(stacked_X, y_test)
    lr_ensemble_prob = best_model.predict_proba(test_X)[:, 1]
    
    rf_accuracy = accuracy_score(y_test, (rf_ensemble_prob> 0.5).astype(int))
    lr_accuracy = accuracy_score(y_test, (lr_ensemble_prob> 0.5).astype(int))
    if rf_accuracy>=lr_accuracy:
        ensemble_prob=rf_ensemble_prob
        ensemble_pred=(ensemble_prob > 0.5).astype(int)
    else:
        ensemble_prob=lr_ensemble_prob
        ensemble_pred=(ensemble_prob > 0.5).astype(int)
    
    print(f"Finall ensemble results of fold_{fold+1} for MACSynDCR:")
    logging.info(f"Finall ensemble results of fold_{fold+1} for MACSynDCR:")
    logging.info("-" * 100)
    print("-" * 100)
    auc = roc_auc_score(y_test, ensemble_prob)
    accuracy = accuracy_score(y_test, ensemble_pred)
    auc_pr = average_precision_score(y_test,ensemble_prob)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    yp=torch.tensor(ensemble_pred.astype(float))
    rmse = np.sqrt(loss_func(y_test.float(), yp))
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc, _ = pearsonr(y_test, ensemble_prob)
            
    print(f'Ensemble Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nEnsemble AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nEnsemble Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Ensemble RMSE: {rmse:.4f} , PCC: {pcc:.4f}')
    logging.info(f'Ensemble Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}, \nEnsemble AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nEnsemble Recall: {recall:.4f}, F1 Score: {f1:.4f}, \nEnsemble RMSE: {rmse:.4f}')
            
    logging.info("-" * 100)
    print("-" * 100)
    return ensemble_pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc
 