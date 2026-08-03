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


from sklearn.utils import class_weight
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, LSTM, Bidirectional
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

import torch
import torch.nn as nn
import torch.optim as optim
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
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

import os
import pickle

import matplotlib.pyplot as plt
import torch
import json
import random

from models.MACSynDCR import Model_ANN
from models.MACSynDCR import MACSynDCR_CNN

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


#batch_size = 256
#epochs = 500
#num_epochs=5
num_classes = 2



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
    
loss_func = nn.MSELoss(reduction='sum')
def Eval_MACSynDCR_ANN(model, train_data,val_data,val_labels,fold):
    print("Evaluation of MACSynDCR_ANN:")
    logging.info("Evaluation of MACSynDCR_ANN model:")
    logging.info("-" * n_delimiter)
    print("-" * n_delimiter)
    vl=val_labels.float()
    model=torch.load(f'saved_models/bestTANN_fold_{fold + 1}.h5')
    #model = torch.load(f'saved_models/bestTANN_fold_{fold + 1}.h5')
    model.eval()
    with torch.no_grad():
            val_outputs = model(val_data).detach().numpy()
            val_predictions = np.argmax(val_outputs,axis=1)
            accuracy = accuracy_score(val_labels, val_predictions)
            AUC = roc_auc_score(val_labels, val_outputs[:,1])
            
            #val_outputs = model(val_data).squeeze()
            train_outputs = model(train_data).detach().numpy()[:,1]
            #val_predictions = (val_outputs > 0.5).float()
            precision = precision_score(val_labels, val_predictions)
            recall = recall_score(val_labels, val_predictions)
            f1 = f1_score(val_labels, val_predictions)
            #mlp_precision, mlp_recall, _ = precision_recall_curve(val_labels.numpy(), val_predictions.numpy())
            #AUC_PR = auc(mlp_precision, mlp_recall)
            AUC = roc_auc_score(val_labels, val_outputs[:,1])
            #print(val_labels.float(), val_predictions.float())
            #print(val_labels.shape,y_pred.shape)
            p=torch.tensor(val_predictions).float()
            #print(vl,p)
            vrmse = np.sqrt(loss_func(vl.float(),p.float() ))
            AUC_PR = average_precision_score(val_labels, val_outputs[:,1])
            pcc, _ = pearsonr(val_labels.float(), val_outputs[:,1])
            
            print(f'Accuracy: {accuracy:.4f}, AUC-ROC: {AUC:.4f} \n AUC-PR: {AUC_PR:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            print(f'\nRMSE: {vrmse:.4f} , PCC: {pcc:.4f}')
            logging.info(f'Accuracy: {accuracy:.4f}, AUC-ROC: {AUC:.4f}, \n AUC-PR: {AUC_PR:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, RMSE: {vrmse:.4f}')
            
    logging.info("-" * n_delimiter)
    print("-" * n_delimiter)
    return val_outputs,train_outputs, AUC, accuracy, AUC_PR, f1, precision, recall, vrmse, pcc



def Eval_MACSynDCR_CNN(model,train_data,val_data,val_labels,fold):
    print("Evaluation of MACSynDCR_CNN:")
    logging.info("Evaluation of MACSynDCR_CNN model:")
    logging.info("-" * n_delimiter)
    print("-" * n_delimiter)
    vl=val_labels.float()
    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    val_data=val_data.reshape(-1,64,38)  
    
    model = torch.load(f'saved_models/CNN_model_fold_{fold + 1}.h5', weights_only=False)
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_data).squeeze()
        val_predictions = (val_outputs > 0.5).astype(float)
        accuracy = accuracy_score(val_labels.numpy(), val_predictions.numpy())
        precision = precision_score(val_labels.numpy(), val_predictions.numpy())
        recall = recall_score(val_labels.numpy(), val_predictions.numpy())
        f1 = f1_score(val_labels.numpy(), val_predictions.numpy())
        AUC_PR = average_precision_score(val_labels, val_outputs)
        AUC = roc_auc_score(val_labels, val_predictions)
        rmse = np.sqrt(loss_func(vl.float(), val_predictions.float()))
        #rmse = np.sqrt(mean_squared_error(val_labels, val_predictions.float()))
        pcc, _ = pearsonr(val_labels, val_predictions.float())
        #vrmse = np.sqrt(loss_func(val_labels.float(), val_predictions.float()))

        print(f'Accuracy: {accuracy:.4f}, AUC-ROC: {AUC:.4f} \n AUC-PR: {AUC_PR:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f'\nRMSE: {rmse:.4f} , PCC: {pcc:.4f}')
            
        logging.info(f'Accuracy: {accuracy:.4f}, AUC-ROC: {AUC:.4f} \n AUC-PR: {AUC_PR:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        logging.info(f'\nRMSE: {rmse:.4f} , PCC: {pcc:.4f}')
                 
    logging.info("-" * n_delimiter)
    print("-" * n_delimiter)
   
    return val_outputs, AUC, accuracy, AUC_PR, f1, precision, recall, rmse, pcc
    
    
    


    
def Eval_MACSynDCR_ANN_model(train_data,val_data,val_labels,fold): 
    print("Evaluation of Keras MACSynDCR_ANN model:")

    scaler = MinMaxScaler()
    val_data = scaler.fit_transform(val_data)
    vl=val_labels.float()
    testy=val_labels
    label_encoder = preprocessing.LabelEncoder()
    y_test = label_encoder.fit_transform(val_labels)

    y_test_cat = to_categorical(y_test,  num_classes = 2)
    
    ANN = load_model('saved_models/best_MACSynDCR_ANN_model.keras')
    loss, accuracy, AUC = ANN.evaluate(val_data, y_test_cat)
    print(f'Evaluate->  Loss: {loss:.4f}, Accuracy: {accuracy:.4f},AUC: {AUC:.4f}')
    #model.save('saved_models/MACSynDCR_ANN_model.hdf5')
    ANN.save(f'saved_models/best_CV_MACSynDCR_ANN_model_Fold_{fold+1}_AUC_{AUC:.3f}_ACC_{accuracy:.3f}.keras')

    pred = ANN.predict(val_data)
    train_pred = ANN.predict(train_data)
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
    #rmse = np.sqrt(mean_squared_error(y_test, ann_pred))
    yp=torch.tensor(y_pred.astype(float))
    rmse = np.sqrt(loss_func(val_labels.float(), yp))
    pcc, _ = pearsonr(y_test_cat, pred)

    print(f'ANN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nANN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nANN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'ANN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    print("*"*100)
    return ann_pred, train_pred
 


def Eval_MACSynDCR_CNN_model(train_data,val_data,val_labels, fold): 
    print("Evaluation of Keras MACSynDCR_CNN model:")

    scaler = MinMaxScaler()
    val_data = scaler.fit_transform(val_data)
    
    testy=val_labels
    label_encoder = preprocessing.LabelEncoder()
    y_test = label_encoder.fit_transform(val_labels)

    y_test_cat = to_categorical(y_test,  num_classes = 2)
    
    val_data = val_data.reshape(-1, 64, 38,1)
    CNN = load_model('saved_models/best_MACSynDCR_CNN_model.keras')
    loss, accuracy, AUC = CNN.evaluate(val_data, y_test_cat)
    print(f'Evaluate->  Loss: {loss:.4f}, Accuracy: {accuracy:.4f},Accuracy: {AUC:.4f}')
    CNN.save(f'saved_models/best_CV_MACSynDCR_CNN_model_Fold_{fold+1}_AUC_{AUC:.3f}_ACC_{accuracy:.3f}.keras')
    #model.save('saved_models/MACSynDCR_ANN_model.hdf5')
    pred = CNN.predict(val_data)
    train_data = train_data.reshape(-1, 64, 38,1)
    train_pred = CNN.predict(train_data)
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
    #rmse = np.sqrt(mean_squared_error(y_test, cnn_pred))
    pcc, _ = pearsonr(y_test_cat, pred)

    print(f'CNN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nCNN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nCNN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'CNN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    print("*"*100)
    return cnn_pred, train_pred
    
    
