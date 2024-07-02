from collections import defaultdict

from scipy.stats import mode

from pyensemble.classify import BaggingEnsembleAlgorithm
from data_distributed import (distributed_single_pruning,
                              distributed_pruning_methods)
from data_distributed import COMEP_Pruning, DOMEP_Pruning

from OPF.meta_ensemble import MetaOPFClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from copy import deepcopy

from time import time
import pickle
import os
import numpy as np
import argparse


np.random.seed(4567)

def get_main_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', type=str, help='Folder where the dataset is located.')
    parser.add_argument('-m', type=str, help='Folder where the models are located.')
    parser.add_argument('-o', type=str, help='Folder where the results will be save.')

    return parser.parse_args()

def fill_missing(lst):
    # Unique values from the informed list
    unique_values = sorted(set(lst))
    
    # Unique values from the sequential values
    new_unique_values = np.arange(len(unique_values))
    
    # 
    for i,v in enumerate(unique_values):
        idx = np.where(np.array(lst) == v)[0]
        for j in idx:
            lst[j] = new_unique_values[i]
    
    return lst

def prediction(models,X_test):
    preds = []

    for m in models:
        y_pred = m.predict(X_test)
        preds.append(y_pred)
    
    return np.array(preds)

if __name__ == '__main__':

    # Gets the command line aguments
    args = get_main_args()
    data = args.i
    models = args.m
    results_folder = args.o

    seed = 11

    # Number of OPFs to compose the ensemble
    n_models = [10,30,50]

    # Number of selected estimators for PyPruning
    n_estimators = [[5],[5,10,15],[10,15,25],[10,30,50]]

    # Pruning models
    pruning_methods = ['COMEP', 'DOMEP']

    # Dataset list
    ds = os.listdir(data)

    # Dictionary to store the metrics of all folds
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # For each dataset
    for d in ds:
        print('Dataset: {}'.format(d))
        
        # For each pruning method
        for p in pruning_methods:

            folds = os.listdir('{}/{}'.format(data,d))
            #folds = [1,2,3]
            
            for f in folds:
                # Loading the training, validation and test sets
                train = np.loadtxt('{}/{}/{}/train.txt'.format(data,d,f),delimiter=',')
                valid = np.loadtxt('{}/{}/{}/valid.txt'.format(data,d,f),delimiter=',')
                test = np.loadtxt('{}/{}/{}/test.txt'.format(data,d,f),delimiter=',')

                # Split the data into X and y
                # The last column is the output variable to be predicted
                # Subtract 1 from the labels so they starts at 0
                X_train,X_valid,X_test = train[:,:-1],valid[:,:-1],test[:,:-1]
                y_train,y_valid,y_test = train[:,-1].astype(int)-1,valid[:,-1].astype(int)-1,test[:,-1].astype(int)-1
                
                # Filling the missing labels
                y = np.concatenate((y_train,y_valid,y_test))
                y = fill_missing(y)

                # Assigns the filled labels to the corresponding sets
                y_train = y[0:len(X_train)]
                y_valid = y[len(X_train):len(X_train)+len(X_valid)]
                y_test = y[len(X_train)+len(X_valid):len(y)]                
                
                # Gets the total number of classes
                classes = np.unique(y)
                n_classes = len(classes)

                for i,n in enumerate(n_models):
                    # Training the ensemble with 'n' OPFs
                    meta_opf_ensemble = MetaOPFClassifier(n_estimators=n,max_samples=None,max_features=None,labels=classes,random_state=seed)
                    meta_opf_ensemble.fit(X_train,y_train)
                    # Models list
                    clfs = np.array([c.classifier for c in meta_opf_ensemble.estimators])

                    # Gets predictions of the base models
                    preds =  prediction(clfs,X_valid)

                    # For each number of estimators to be selected
                    for ne in n_estimators[i]:                        
                        output_folder = '{}/EPFD_{}/{}/{}_estimators_{}/pruning/{}'.format(results_folder,p,d,n,ne,f)

                        if (os.path.exists('{}/metrics.txt'.format(output_folder))):
                            print('Folder {} with metrics already exists. Going to the next iteration...'.format(output_folder))
                            continue

                        if (not os.path.exists(output_folder)):
                            os.makedirs(output_folder)
                        
                        # Pruning the ensemble model
                        start_time = time()
                        if 'COMEP' in p:
                            idx = COMEP_Pruning(np.array(preds).T.tolist(), ne, y_valid, 0.5)
                        elif 'DOMEP' in p:
                            idx = DOMEP_Pruning(np.array(preds).T.tolist(), ne, 2, y_valid, 0.5)
                        end_time = time() - start_time

                        pruned = clfs[idx]

                        y_pred = mode(prediction(pruned,X_test))[0].flatten()

                        print('y_pred: ',y_pred)
                        print('y_test: ',y_test)

                        # Computing the metrics
                        acc = accuracy_score(y_test,y_pred)
                        precision = precision_score(y_test,y_pred,average='weighted')
                        recall = recall_score(y_test,y_pred,average='weighted')
                        f1 = f1_score(y_test,y_pred,average='weighted')

                        # Adds each fold iteration to the global dictionary
                        all_metrics[p][d]['{}_estimators_{}'.format(n,ne)][f] = np.array([acc,precision,recall,f1,len(pruned)])
                        
                        # Saving the validation measures, the meta_X and y_pred
                        np.savetxt('{}/y_pred.txt'.format(output_folder),y_pred,fmt='%.4f',delimiter=',')
                        np.savetxt('{}/y_test.txt'.format(output_folder),y_test,fmt='%.4f',delimiter=',')
                        np.savetxt('{}/metrics.txt'.format(output_folder),np.array([acc,precision,recall,f1,len(pruned)]),fmt='%.4f',delimiter=',',header='Accuracy,precision,recall,F1,n_estimators')
    
    # Computes the mean for each fold
    for p in all_metrics:
        # For each dataset
        for d in all_metrics[p]:
            # For each number of estimators
            for n in all_metrics[p][d]:
                metrics = []
                # For each fold
                for f in all_metrics[p][d][n]:
                    metrics.append(all_metrics[p][d][n][f])
                
                np.savetxt('{}/{}/{}/{}/pruning/all_metrics.txt'.format(results_folder,p,d,n),metrics,header='Accuracy,precision,recall,F1,n_estimators')
        
    print('BASELINES EXECUTION FINISHED SUCCESSFULLY!')