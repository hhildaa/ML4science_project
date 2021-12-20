import pandas as pd
import torch
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
import params
from model import train, FeedForward, labels_to_one_hot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC
import numpy as np
from accuracy import *
from cross_validation import *
#from imblearn.over_sampling import SMOTENC

######################################################################################################################

# set random seed for REPRODUCIBILITY
torch.manual_seed(params.SEED)
np.random.seed(params.SEED)
random.seed(params.SEED)

######################################################################################################################

# Reading the data

if params.PREPROCESS and params.BIN_AGES:
    input_data = pd.read_csv('dataset/age_binned_preprocessed.csv', sep='\t', index_col=0)
elif params.PREPROCESS:
    input_data = pd.read_csv('dataset/preprocessed_data.csv', sep='\t', index_col=0)
elif params.BIN_AGES:
    input_data = pd.read_csv('dataset/age_binned.csv', sep='\t', index_col=0)
else:
    input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t', index_col=0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.TEST_FRACTION)
train_df = input_data.sample(frac=0.8, random_state=100)
test_df = pd.concat([input_data, train_df, train_df]).drop_duplicates(keep=False)

# upsampling minority classes
if params.UPSAMPLING:
    if params.UPSAMPLING_TYPE == "SMOTE":
        X_train, y_train = train_df.drop(columns=['severity']), train_df['severity']

        categorical_features = range(2,34)
        smote_nc = SMOTENC(categorical_features=categorical_features, random_state=100)
        X_train, y_train = smote_nc.fit_resample(X_train, y_train)
    elif params.UPSAMPLING_TYPE == "random":
        class_values = list(train_df['severity'].unique())
        class_lens = [train_df[train_df['severity'] == class_val].shape[0] for class_val in class_values]
        max_len = max(class_lens)

        for class_val in class_values:
            class_df = train_df[train_df['severity'] == class_val]
            if class_df.shape[0] < max_len:
                to_add = class_df.sample(max_len - class_df.shape[0], replace=True)
                train_df = pd.concat([train_df, to_add], axis=0)

        # mix randomly
        train_df = train_df.sample(frac=1)
        print(train_df['severity'].value_counts())
        
        X_train, y_train = train_df.drop(columns=['severity']), train_df['severity']
    else:
        raise ValueError('Wrong upsampling type selected')
else:
    X_train, y_train = train_df.drop(columns=['severity']), train_df['severity']
 
X_test, y_test = test_df.drop(columns=['severity']), test_df['severity']

print(X_train.head())

######################################################################################################################
top_accuracy = -1
top_params = None
for hidden_size in params.GRID_SEARCH_PARAMS['hidden_sizes']:
    for learning_rate in params.GRID_SEARCH_PARAMS['learning_rates']:
        for num_epochs in params.GRID_SEARCH_PARAMS['epochs']:
            # Training on train data
            if params.MODEL_TYPE == 'FeedForward':
                model = FeedForward(hidden_size, params.PMF_LAYER, params.PMF_TYPE)

                # Loss functions
                loss_func_mse = nn.MSELoss()
                loss_func_mae = nn.L1Loss()
                loss_func_cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]))

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                #model, train_loss = train(X_train.to_numpy(), y_train.to_numpy(), model, loss_func_cross_entropy, optimizer, params.NUM_EPOCHS)

                best_accuracy, cur_acc_test, dcas, ampcas, gmpcas, qwks = k_fold_cross_validation(y_train, X_train, params.K_FOLDS, model, loss_func_cross_entropy, optimizer, num_epochs, hidden_size)
                if best_accuracy > top_accuracy:
                    top_accuracy = best_accuracy
                    top_params = (hidden_size, learning_rate, num_epochs)

            elif params.MODEL_TYPE == 'LinearRegression':
                model = LinearRegression()
                model = model.fit(X_train, y_train)
                
                y_pred = np.round_(model.predict(X_test))

                print(y_pred)
                print(accuracy_score(y_test, y_pred))

            elif params.MODEL_TYPE == 'SVC':
                model = SVC()
                model = model.fit(X_train, y_train)

                y_pred = np.round_(model.predict(X_test))

                print(y_pred)
                print(accuracy_score(y_test, y_pred))

print(top_params, top_accuracy)

######################################################################################################################

# Testing on test data

#test_estimation, train_acc, acc, dca, ampca, gmpca, qwk = test(model, X_train, y_tr