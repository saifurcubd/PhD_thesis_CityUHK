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

import argparse
import os
import time
import pickle
import logging


import numpy as np
import pandas as pd
import csv
import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import pickle
import json
import random

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



#from models.MACSynDCR import DrugSynergyCNN,Model_ANN
#from models.MACSynDCR import MACSynDCR_CNN,MACSynDCR_LSTM
#from models.MACSynDCR_evaluation import Eval_MACSynDCR_CNN, Eval_MACSynDCR_LSTM

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
learning_rate = 0.0001

class MACSynDCR_ANN(nn.Module):
    def __init__(self):
        super(MACSynDCR_ANN, self).__init__()
        self.fc1 = nn.Linear(2688, 1024)
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


def MACSynDCR_ANN_model():
    model = keras.Sequential([
        layers.Input(shape=(2688,)),
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
    model1.add(Conv2D(32, (3, 3), input_shape = (64,42,1), activation='relu'))
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






def Train_and_Validation_MACSynDCR_ANN(X_train,X_test,y_train,y_test,args):
    print("Data processing and Training of MACSynDCR_ANN model:")
    num_epochs =args.epoch
    batch_size = 256
    learning_rate = 0.0001
    #print(X_train.shape)
    #print(X_test.shape)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)
   


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

    print(f'ANN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nANN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nANN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'\ANN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    return ann_pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc[1]
    

def Train_and_Validation_MACSynDCR_CNN(X_train,X_test,y_train,y_test,args):
    print("Data processing and Training of MACSynDCR_CNN model:")
    num_epochs = args.epoch
    batch_size = 256
    learning_rate = 0.0001
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.fit_transform(X_test)
    #print(x_train.shape)
    #print(x_test.shape)


    #label encoding y_train
    testy=y_test
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    y_test_cat = to_categorical(y_test,  num_classes = 2)
    y_train_cat = to_categorical(y_train, num_classes = 2)
   # print(y_test_cat.shape, y_train_cat.shape)
    
    x_train = x_train.reshape(-1, 64, 42,1)
    x_test = x_test.reshape(-1, 64, 42,1)
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

    print(f'CNN: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \nCNN: AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, \nCNN Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'CNN:RMSE: {rmse:.4f} , PCC: {pcc[1]:.4f}')
    return cnn_pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc[1]


def Train_MACSynDCR_ANN(train_data, train_labels,val_data,val_labels, args,fold):
        print("Data processing and Training of MACSynDCR_ANN model:")
        
        num_epochs=int(args.epoch)
        
        
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.fit_transform(val_data)
        
        train_data=torch.tensor(train_data).float()
        val_data=torch.tensor(val_data).float()
       
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
        return model, best_preds,train_preds     
        
        
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
    
    val_data = val_data.reshape(-1, 64, 42,1)
    CNN = load_model('saved_models/best_MACSynDCR_CNN_model.keras')
    loss, accuracy, AUC = CNN.evaluate(val_data, y_test_cat)
    print(f'Evaluate->  Loss: {loss:.4f}, Accuracy: {accuracy:.4f},Accuracy: {AUC:.4f}')
    CNN.save(f'saved_models/best_CV_MACSynDCR_CNN_model_Fold_{fold+1}_AUC_{AUC:.3f}_ACC_{accuracy:.3f}.keras')
    #model.save('saved_models/MACSynDCR_ANN_model.hdf5')
    pred = CNN.predict(val_data)
    train_data = train_data.reshape(-1, 64, 42,1)
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


from tensorflow.keras.optimizers import RMSprop,Adam, SGD
from torchsummary import summary
#from models.MACSynDCR_model import Ensemble_MACSynDCR_model
from sklearn.linear_model import LogisticRegression


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

def Ensemble_MACSynDCR_model(out_dir,ann_preds,kann_preds, kcnn_preds,y_test, fold,val_index): 
   
    ann_auc = roc_auc_score(y_test, ann_preds)
    ann_accuracy = accuracy_score(y_test, (ann_preds > 0.5).astype(int))
    print(f'Accuracy Score and AUC for ANN = AUC: {ann_auc:.4f}, Accuracy: {ann_accuracy:.4f}')
    
    kann_auc = roc_auc_score(y_test, kann_preds)
    kann_accuracy = accuracy_score(y_test, (kann_preds > 0.5).astype(int))
    print(f'Accuracy Score and AUC for KANN = AUC: {kann_auc:.4f}, Accuracy: {kann_accuracy:.4f}')
    
    kcnn_auc = roc_auc_score(y_test, kcnn_preds)
    kcnn_accuracy = accuracy_score(y_test, (kcnn_preds> 0.5).astype(int))
    print(f'Accuracy Score and AUC for KCNN = AUC: {kcnn_auc:.4f}, Accuracy: {kcnn_accuracy:.4f}')
    
    
    test_X = np.column_stack((ann_preds,kann_preds,kcnn_preds))
    #print(train_X.shape, type(test_X.shape))
    
    avg_pred=np.mean(test_X, axis=1) 
    #mean_pred = np.mean(ensemble_pred, axis=1)
    df1 = pd.DataFrame({
    'sample_id': val_index,
    'probability': avg_pred
    })
    csv_file1 = os.path.join(out_dir, f'avg_prob{fold+1}.csv')
    df1.to_csv(csv_file1, index=False)
    
    max_pred=np.max(test_X, axis=1)
    df2 = pd.DataFrame({
    'sample_id': val_index,
    'probability': max_pred
    })

    csv_file2 = os.path.join(out_dir, f'max_prob{fold+1}.csv')
    df2.to_csv(csv_file2, index=False)
    
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
    param_grid = {'C': [0.01,0.1, 1, 10,100,1000,10000], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5,verbose=1)
    grid_search.fit(test_X, y_test)
    best_model = grid_search.best_estimator_
    #meta_model = LogisticRegression()
    #meta_model.fit(stacked_X, y_test)
    lr_ensemble_prob = best_model.predict_proba(test_X)[:,1]
    ensemble_prob=lr_ensemble_prob
    ensemble_pred=(ensemble_prob > 0.5).astype(int)
    df3 = pd.DataFrame({
    'sample_id': val_index,
    'probability': ensemble_pred
    })
    csv_file3 = os.path.join(out_dir, f'ensemble_prob{fold+1}.csv')
    df3.to_csv(csv_file3, index=False)
    
    
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
    return ensemble_prob, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc


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
    


time_str = str(datetime.now().strftime('%y%m%d%H%M'))

#Training and Evaluation Function

input_size=2688
input_shape = (64, 42) 
loss_func = nn.MSELoss(reduction='sum')
from sklearn.model_selection import StratifiedKFold

def train_and_evaluate_model(data, labels, args, out_dir):
    
    save_args(args, os.path.join(out_dir, 'args.json'))
    test_loss_file = os.path.join(out_dir, 'test_loss.pkl')
    
    num_epochs=args.epoch 
    learning_rate=args.lr[1]
    
    #kfold=StratifiedKFold(n_splits=5, *, shuffle=True, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_accuracy = 0.0
    
    n_delimiter = 100
    
    
    fauc=[]
    facc=[]
    faucpr=[]
    ff1=[]
    fprec=[]
    frecall=[]
    frmse= []
    fpcc=[]
    
    pred_results = {
        'ann_pred': [],
        'cnn_pred': [],
        'lstm_pred': [],
        'Precision': [],
        'kann_pred': [],
        'kcnn_pred': [],
        'klstm_pred': [],
    }
    
    for fold, (train_index, val_index) in enumerate(kfold.split(data)):
        print(f'Fold {fold + 1}')
        logging.info(f'Fold {fold + 1}')
        logging.info("-" * n_delimiter)
        
        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        
        ANNmodel, best_preds, train_preds=Train_MACSynDCR_ANN(train_data, train_labels,val_data,val_labels, args,fold)
      
        val_outputs = best_preds.astype(float)
        val_predictions = (val_outputs > 0.5).astype(float)
        #print(val_labels.numpy(),val_predictions,val_outputs)
        accuracy = accuracy_score(val_labels.numpy(), val_predictions)
        precision = precision_score(val_labels.numpy(), val_predictions)
        recall = recall_score(val_labels.numpy(), val_predictions)
        f1 = f1_score(val_labels.numpy(), val_predictions)
        auc_pr = average_precision_score(val_labels.float(), val_outputs.astype(float))
        auc = roc_auc_score(val_labels.float(), val_outputs.astype(float))
        rmse = np.sqrt(loss_func(val_labels.float(),torch.FloatTensor(val_outputs.squeeze())))
        #rmse = np.sqrt(mean_squared_error(val_labels, val_predictions.float()))
        #print(val_labels, torch.FloatTensor(val_outputs.squeeze()))
        pcc, _ = pearsonr(val_labels, torch.FloatTensor(val_outputs.squeeze()))
       # pcc, a = pearsonr([1,1,1], [1,1,4])
        #vrmse = np.sqrt(loss_func(val_labels.float(), val_predictions.float()))
        print(f'Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f} \n AUC-PR: {auc_pr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f'RMSE: {rmse:.4f} , PCC: {pcc:.4f}')
        ann_pred=val_outputs
        ann_train_pred=train_preds
        
        Train_and_Validation_MACSynDCR_ANN(train_data,val_data,train_labels,val_labels,args)
        Train_and_Validation_MACSynDCR_CNN(train_data,val_data,train_labels,val_labels,args )
        
       
        
        train_pred=[]
        #Eval_MACSynDCR_CNN(CNNmodel, train_data,val_data,val_labels,fold)
        #ann_pred, ann_train_pred,auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc=Eval_MACSynDCR_ANN(ANNmodel, train_data,val_data,val_labels,fold)
        kann_pred, kann_train_pred=Eval_MACSynDCR_ANN_model(train_data,val_data,val_labels,fold)
        kcnn_pred, kcnn_train_pred=Eval_MACSynDCR_CNN_model(train_data,val_data,val_labels,fold)
        train_pred.append(ann_train_pred)
        train_pred.append(kann_train_pred)
        train_pred.append(kcnn_train_pred)

        ensemble_pred, auc, accuracy, auc_pr, f1, precision, recall, rmse, pcc=Ensemble_MACSynDCR_model(out_dir,ann_pred,kann_pred, kcnn_pred,val_labels, fold,val_index)
        fauc.append(auc)
        facc.append(accuracy)
        faucpr.append(auc_pr)
        ff1.append(f1)
        fprec.append(precision)
        frecall.append(recall)
        frmse.append(rmse)
        fpcc.append(pcc)
        #ensemble_prob=np.array(ensemble_prob)
        
 
    
    #mu, sigma = calc_stat(test_losses)
    print("*"*n_delimiter)
    logging.info("*" * n_delimiter)
    print("*           Final MACSynCDR Results:")
    logging.info("*          Final MACSynCDR Results:")
    logging.info("*" * n_delimiter)
    print("*"*n_delimiter)
    mu, sigma = calc_stat(fauc)
    print(" AUC: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" AUC: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(facc)
    print(" Accuracy: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" Accuracy: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(faucpr)
    print(" AUC-PR: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" AUC-PR: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(fprec)
    print(" Precision: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" Precision: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(frecall)
    print(" Recall: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" Recall: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(ff1)
    print(" F1: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" F1: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(frmse)
    print(" RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
    
    mu, sigma = calc_stat(fpcc)
    print(" PCC: {:.4f} ± {:.4f}".format(mu, sigma))
    logging.info(" PCC: {:.4f} ± {:.4f}".format(mu, sigma))
    
    print("Max AUC: {:.4f}".format(max(fauc)))
    print("Max ACC: {:.4f}".format(max(facc)))
    print("Max AUC-PR: {:.4f}".format(max(faucpr)))
    print("Max Precision: {:.4f}".format(max(fprec)))
    print("Max Reacall: {:.4f}".format(max(frecall)))
    print("Max F1: {:.4f}".format(max(ff1)))
    print("MIN RMSE: {:.4f}".format(min(frmse)))
    print("Max PCC: {:.4f}".format(max(fpcc)))
    
    return best_model
    
  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help="n epoch")
    parser.add_argument('--batch', type=int, default=256, help="batch size")
    #parser.add_argument('--gpu', type=int, default=None, help="cuda device")
    #parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
    parser.add_argument('--suffix', type=str, default=time_str, help="model dir suffix")
    #parser.add_argument('--hidden', type=int, nargs='+', default=[2048, 4096, 8192], help="hidden size")
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5], help="learning rate")
    args = parser.parse_args()
    out_dir = os.path.join(OUTPUT_DIR, 'cv_{}'.format(args.suffix))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'cv.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)

    # Convert DataFrame to NumPy
    #dataset
    #drugdata = pd.read_csv('OneilProcessDataset30_dti.csv')
    #drugdata = pd.read_csv('OneilProcessDataset_0to10_2176_dti.csv')
    #drugdata = pd.read_csv('OneilProcessDataset_20to20_2176_dti.csv')
    #drugdata = pd.read_csv('OneilProcessDataset_all_2432_d1d2.csv')
    #drugdata = pd.read_csv('OneilProcessDataset_2176_d1cd2.csv')
    #drugdata = pd.read_csv('MFSynDCP_ProcessDataset_2176_d1cd2-10to10-Z.csv')
    drugdata = pd.read_csv('SDDSynergyProcessDataset_0to10_2688_dti_LMGAN.csv')
    drugdata = pd.DataFrame(data=drugdata)
    xdata=drugdata.iloc[:,1:2689]
    ydata=drugdata.iloc[:,2689]
    Y = ydata
    X = xdata
    Y.value_counts()
    #xdata=to01(xdata.values)
    
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = DecisionTreeClassifier(random_state=42)
 
    data = torch.FloatTensor(xdata.values)
    labels = torch.FloatTensor(ydata.values)

    best_model = train_and_evaluate_model(data, labels,args, out_dir) ##CV

if __name__ == "__main__":
    main() 
    