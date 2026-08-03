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

from sklearn.model_selection import train_test_split, KFold

#from models.MACSynDCR import DrugSynergyCNN,Model_ANN
#from models.MACSynDCR import MACSynDCR_CNN,MACSynDCR_LSTM
#from models.MACSynDCR_evaluation import Eval_MACSynDCR_CNN, Eval_MACSynDCR_LSTM

from models.MACSynDCR_training import Train_and_Validation_MACSynDCR_ANN
from models.MACSynDCR_training import Train_and_Validation_MACSynDCR_CNN

from models.MACSynDCR_training import Train_MACSynDCR_ANN
from models.MACSynDCR_training import Train_MACSynDCR_CNN

from models.MACSynDCR_evaluation import Eval_MACSynDCR_ANN, Eval_MACSynDCR_CNN#, Eval_MACSynDCR_ANN
from models.MACSynDCR_evaluation import Eval_MACSynDCR_ANN_model, Eval_MACSynDCR_CNN_model

from tensorflow.keras.optimizers import RMSprop,Adam, SGD
from torchsummary import summary
#from models.MACSynDCR_model import Ensemble_MACSynDCR_model
from sklearn.linear_model import LogisticRegression

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
    'probability': ensemble_prob
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

input_size=2432
input_shape = (64, 38) 
loss_func = nn.MSELoss(reduction='sum')
from sklearn.model_selection import StratifiedKFold

def train_and_evaluate_model(data, labels, args, out_dir):
    
    save_args(args, os.path.join(out_dir, 'args.json'))
    test_loss_file = os.path.join(out_dir, 'test_loss.pkl')
    
    num_epochs=args.epoch 
    learning_rate=args.lr[0]
    
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
        
        ANNmodel, best_preds, train_preds=Train_MACSynDCR_ANN(train_data, train_labels,val_data,val_labels, args,fold,val_data)
      
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
        
        Train_and_Validation_MACSynDCR_ANN(train_data,val_data,train_labels,val_labels,args,val_data)
        Train_and_Validation_MACSynDCR_CNN(train_data,val_data,train_labels,val_labels,args,val_data)
        
       
        
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
    drugdata = pd.read_csv('MACSynDCRP_ProcessData_m10to20.csv')
    drugdata = pd.DataFrame(data=drugdata)
    xdata=drugdata.iloc[:,1:2433]
    ydata=drugdata.iloc[:,2433]
    Y = ydata
    X = xdata
    Y.value_counts()
    #xdata=to01(xdata.values)
    
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

# -------- Step 1: 80-20 Split --------
  #  X_trainval, X_test, y_trainval, y_test = train_test_split(
   # X, Y, test_size=0.2, random_state=42)


 
    data = torch.FloatTensor(xdata.values)
    labels = torch.FloatTensor(ydata.values)

    best_model = train_and_evaluate_model(data, labels,args, out_dir) ##CV

if __name__ == "__main__":
    main() 