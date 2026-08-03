import numpy as np
import pandas as pd
import argparse
import os
import time
import pickle
import logging
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
from keras.models import load_model


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

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



from models.MACSynDCR_model import MACSynDCR_ANN, MACSynDCR_CNN, Ensemble_MACSynDCR_model
from models.MACSynDCR_model import MACSynDCR_ANN_model,MACSynDCR_CNN_model



import warnings
warnings.filterwarnings("ignore")


from datetime import datetime

def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma
    
def save_args(args, save_to: str):
    args_dict = args.__dict__
    with open(save_to, 'w') as f:
        json.dump(args_dict, f, indent=2)


OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
n_delimiter = 100

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

#num_epochs=1
learning_rate=0.001

loss_func = nn.MSELoss(reduction='sum')

def Train_MACSynDCR_ANN(train_data, train_labels,val_data,val_labels, args,fold,ind_test):
        print("Data processing and Training of MACSynDCR_ANN model:")
        
        num_epochs=int(args.epoch)
        
        
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.fit_transform(val_data)
        ind_test = scaler.fit_transform(ind_test)
        
        train_data=torch.tensor(train_data).float()
        val_data=torch.tensor(val_data).float()
        ind_test=torch.tensor(ind_test).float()
       
        #label encoding y_train
        testy=val_labels
        label_encoder = preprocessing.LabelEncoder()
        y_train = label_encoder.fit_transform(train_labels)
        y_test = label_encoder.fit_transform(val_labels)
        
        

        y_test_cat = to_categorical(y_test,  num_classes = 2)
        y_train_cat = to_categorical(y_train, num_classes = 2)
        #print(y_test_cat.shape, y_train_cat.shape)
        
        model=MACSynDCR_ANN()        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        #MSE loss
        loss_func = nn.MSELoss(reduction='sum')
        savefile="yes"
        model.train()
        best_val_acc=0.0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(train_data)#.detach().numpy()  # Ensure correct shape
            #print(outputs, outputs.shape)
            #v=np.argmax(outputs,axis=1).astype(float)
            #t=torch.FloatTensor(v)
            loss = criterion(outputs, train_labels.long())
            #trmse = np.sqrt(mean_squared_error(train_labels.float(), outputs.detach().numpy()))
            val_outputs = model(val_data).detach().numpy()
            val_predictions = np.argmax(val_outputs,axis=1)
            accuracy = accuracy_score(val_labels, val_predictions)
            AUC = roc_auc_score(val_labels, val_outputs[:,1])
            
            if accuracy > best_val_acc:
                best_preds=0.0
                best_val_acc = accuracy
                best_preds=val_outputs[:,1]
                #print(best_preds)
                train_preds=0.0
                train_pr= model(train_data).detach().numpy()
                train_preds=train_pr[:,1]
                
                ipreds=0.0
                i_pr= model(ind_test).detach().numpy()
                ipreds=i_pr[:,1]
                
                #best_model = model.state_dict()
                #torch.save(best_model, f'best_model_fold_{fold + 1}.pt')
                torch.save(model, f'saved_models/bestTANN_fold_{fold + 1}.h5')
                torch.save(model, f'saved_models/ANN_model_fold_{fold + 1}_ACC_{accuracy:.5f}_AUC_{AUC:.5f}.h5')
                logging.info(f"ANN_model saved with Val_ACC: {accuracy:.4f}")
                print(f"ANN_model saved with Val_ACC: {accuracy:.4f} Val_AUC: {AUC:.4f}")
                logging.info("-" * n_delimiter)
                if savefile!="yes":
                    os.remove(savefile)
                savefile=f'saved_models/ANN_model_fold_{fold + 1}_ACC_{accuracy:.5f}_AUC_{AUC:.5f}.h5'
            
            loss.backward()
            optimizer.step()
            if (epoch%20)==0:
                print(f'fold_{fold+1}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val_ACC: {accuracy:.4f}, Val_AUC: {AUC:.4f}')
                logging.info(f'fold_{fold+1}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, ACC: {accuracy:.4f}, AUC: {AUC:.4f}')
        return model, best_preds,train_preds#,ipreds     
        
        




def Train_MACSynDCR_CNN(X,y,val_data,val_labels,args,fold):
    print("Data processing and Training of MACSynDCR_CNN model:")
    batch_size = 128
    lr=0.001
    num_epochs=int(args.epoch/2)+1
    
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32)
    X_train_tensor=X_train_tensor.reshape(-1,64,38)
    
    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    val_data=val_data.reshape(-1,64,38)
    
   # print(X_train_tensor.shape)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       
    model = MACSynDCR_CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    best_val_acc=0.0
    savefile="yes"
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        val_outputs = model(val_data).squeeze()
        val_predictions = (val_outputs > 0.5).float()
        accuracy = accuracy_score(val_labels.numpy(), val_predictions.numpy())
        AUC = roc_auc_score(val_labels, val_predictions)
        if accuracy > best_val_acc:
                best_val_acc = accuracy
                #best_model = model.state_dict()
                #torch.save(best_model, f'best_model_fold_{fold + 1}.pt')
                torch.save(model, f'saved_models/CNN_model_fold_{fold + 1}_{accuracy:.4f}.h5')
                torch.save(model, f'saved_models/CNN_model_fold_{fold + 1}.h5')
                logging.info(f"CNN_model saved with Val_ACC: {accuracy:.4f}")
                print(f"CNN_model saved with Val_ACC: {accuracy:.4f}")
                logging.info("-" * n_delimiter)
                if savefile!="yes":
                    os.remove(savefile)
                savefile=f'saved_models/CNN_model_fold_{fold + 1}_{accuracy:.4f}.h5'
        if epoch%10==0:
            print(f'CNN fold_{fold+1}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val ACC: {accuracy:.4f}, , Val AUC: {AUC:.4f}')
            logging.info(f'CNN fold_{fold+1}: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val ACC: {accuracy:.4f}, ,Val AUC: {AUC:.4f}')
        
    return model
    
    

    
#Keras models:    
    
def Train_and_Validation_MACSynDCR_ANN(X_train,X_test,y_train,y_test,args,intTestData):
    print("Data processing and Training of MACSynDCR_ANN model:")
    num_epochs =args.epoch
    batch_size = 128
    learning_rate = 0.001
    #print(X_train.shape)
    #print(X_test.shape)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)
    
    intTestData = scaler.fit_transform(intTestData)
   


    #label encoding y_train
    testy=y_test
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    y_test_cat = to_categorical(y_test,  num_classes = 2)
    y_train_cat = to_categorical(y_train, num_classes = 2)
    #print(y_test_cat.shape, y_train_cat.shape)
    

    model=MACSynDCR_ANN_model()
    model.save(f'saved_models/best_MACSynDCR_ANN_model.keras')
    model.compile(loss ='categorical_crossentropy', optimizer='adam',metrics =['acc','auc'])
    checkpoint_ann = ModelCheckpoint(
        'saved_models/best_MACSynDCR_ANN_model.keras',  # Filepath to save the model
        monitor='val_auc',  # Monitor validation accuracy
        save_best_only=True,  # Save only the best model
        mode='max',  # Maximize the monitored value
        verbose=1)
              
    os.remove(f'saved_models/best_MACSynDCR_ANN_model.keras')
    model.fit(x_train, y_train_cat, epochs=num_epochs, batch_size=batch_size,validation_data=(x_test, y_test_cat),callbacks=[checkpoint_ann])
    # Load the Keras model
    ANN = load_model('saved_models/best_MACSynDCR_ANN_model.keras')
    #model.save('saved_models/MACSynDCR_ANN_model.hdf5')
    pred = ANN.predict(x_test)
    ann_pred=pred[:,1]
    ann_label=np.argmax(pred,axis=1)
    y_pred_prob=ann_pred
    y_pred=ann_label
    auc = roc_auc_score(y_test_cat, pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test_cat,pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    yp=torch.tensor(y_pred.astype(float))
    rmse = np.sqrt(loss_func(testy.float(), yp))
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc, _ = pearsonr(y_test_cat, pred)
    
    ipred = ANN.predict(intTestData)
    ann_ipred=ipred[:,1]

    print(f'ANN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nANN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nANN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'\ANN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    return ann_ipred
    


def Train_and_Validation_MACSynDCR_Ensemble(X_train,X_test,y_train,y_val,args):
    print("Data processing and Training of MACSynDCR_CNN model:")
    num_epochs = args.epoch
    batch_size = 128
    learning_rate = 0.001
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)
    
    #intTestData = scaler.fit_transform(intTestData)
    
    model=Model5()
    history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_val),
                    epochs=num_epochs, 
                    batch_size=128)

    X_train = x_train.reshape(-1, 64, 38)
    X_val = x_test.reshape(-1, 64, 38)
    model=MACSynDCR_Hybrid()
    model.compile(loss ='binary_crossentropy', optimizer='adam',metrics =['acc','auc'])
    checkpoint_ensemble = ModelCheckpoint(
        'saved_models/best_MACSynDCR_Ensemble_model.keras',  # Filepath to save the model
        monitor='val_auc',  # Monitor validation accuracy
        save_best_only=True,  # Save only the best model
        mode='max',  # Maximize the monitored value
        verbose=1)
    history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=num_epochs, 
                    batch_size=128,
                    callbacks=[checkpoint_ensemble])

    Ensemble = load_model('saved_models/best_MACSynDCR_Ensemble_model.keras')
    # Evaluate the model on validation data
    loss, accuracy, AUC = Ensemble.evaluate(X_val, y_val)
    print(f'Evaluate->  Loss: {loss:.4f}, Accuracy: {accuracy:.4f},Accuracy: {AUC:.4f}')
   
    pred = Ensemble.predict(X_val).flatten()
    
    print(pred.shape,pred)
    #lstm_pred=pred[:,1]
    #ylabel=np.argmax(pred,axis=1)
    #y_pred_prob=cnn_pred
    #print(type(y_val),type(pred))
    y_test=y_val
    y_pred=(pred> 0.5).astype(int)
    auc = roc_auc_score(y_test, pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test,pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    #rmse = np.sqrt(mean_squared_error(y_test, pred))
    y_test=np.array(y_val).astype(float)
    #print(y_test.shape)
    #print(pred.shape)
    val_outputs = Ensemble.predict(X_val).squeeze()
    val_predictions = (val_outputs > 0.5).astype(float)
    rmse = np.sqrt(loss_func(y_test, val_predictions))

    #rmse = np.sqrt(mean_squared_error(train_labels.float(), outputs.detach().numpy()))

    pcc, _ = pearsonr(y_test, pred)
    #print(f'LSTM2: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nLSTM: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nLSTM2 Recall: {recall:.4f}, F1 Score: {f1:.4f},\nLSTM:RMSE: {rmse:.4f} , PCC: {pcc:.4f}')
    #print(f'\nLSTM:RMSE: {rmse:.4f} ')
    with np.printoptions(precision=4, suppress=True):
        print(f'Ensemble: Accuracy: {accuracy}, AUC-ROC: {auc} \nLSTM: AUC-PR: {auc_pr}, Precision: {precision}, \nLSTM2 Recall: {recall}, F1 Score: {f1},\nLSTM:RMSE: {rmse} , PCC: {pcc}')
    return pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc[1]


# For keras


def Train_and_Validation_MACSynDCR_CNN(X_train,X_test,y_train,y_test,args,intTestData):
    print("Data processing and Training of MACSynDCR_CNN model:")
    num_epochs = args.epoch
    batch_size = 128
    learning_rate = 0.001
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)
    #print(x_train.shape)
    #print(x_test.shape)
    intTest = scaler.fit_transform(intTestData)

    #label encoding y_train
    testy=y_test
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    y_test_cat = to_categorical(y_test,  num_classes = 2)
    y_train_cat = to_categorical(y_train, num_classes = 2)
   # print(y_test_cat.shape, y_train_cat.shape)
    
    x_train = x_train.reshape(-1, 64, 38,1)
    x_test = x_test.reshape(-1, 64, 38,1)
    model=MACSynDCR_CNN_model()
    model.save(f'saved_models/best_MACSynDCR_CNN_model.keras')
    model.compile(loss ='categorical_crossentropy', optimizer='adam',metrics =['acc','auc'])
    checkpoint_cnn = ModelCheckpoint(
        'saved_models/best_MACSynDCR_CNN_model.keras',  # Filepath to save the model
        monitor='val_acc',  # Monitor validation accuracy
        save_best_only=True,  # Save only the best model
        mode='max',  # Maximize the monitored value
        verbose=1)
              
    os.remove(f'saved_models/best_MACSynDCR_CNN_model.keras')
    model.fit(x_train, y_train_cat, epochs=num_epochs, batch_size=batch_size,validation_data=(x_test, y_test_cat),callbacks=[checkpoint_cnn])
    # Load the Keras model
    CNN = load_model('saved_models/best_MACSynDCR_CNN_model.keras')
    #model.save('saved_models/MACSynDCR_CNN_model.hdf5')
    pred = CNN.predict(x_test)
    #print(pred.shape,pred)
    cnn_pred=pred[:,1]
    cnn_label=np.argmax(pred,axis=1)
    y_pred_prob=cnn_pred
    y_pred=cnn_label
    auc = roc_auc_score(y_test_cat, pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test_cat,pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    yp=torch.tensor(y_pred.astype(float))
    rmse = np.sqrt(loss_func(testy.float(), yp))
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pcc, _ = pearsonr(y_test_cat, pred)
    
    intTest = intTest.reshape(-1, 64, 38,1)
    ipred = CNN.predict(intTest)
    #print(pred.shape,pred)
    cnn_ipred=ipred[:,1]

    print(f'CNN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nCNN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nCNN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'CNN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    return cnn_ipred
    


    
    

    


 



        