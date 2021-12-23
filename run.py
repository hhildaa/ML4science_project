import random
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC
import numpy as np
#from imblearn.over_sampling import SMOTEN, SMOTENC

from model import train, FeedForward
from evaluation import *
from cross_validation import *
import params


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

train_df = input_data.sample(frac=0.8, random_state=100)
test_indices = set(input_data.index) - set(train_df.index)
test_df = input_data.loc[test_indices, :]

# upsampling minority classes
if params.UPSAMPLING:

    if params.UPSAMPLING_TYPE == "SMOTE":
        X_train, y_train = train_df.drop(columns=['severity']), train_df['severity']
        if params.BIN_AGES:
            smote_n = SMOTEN(random_state=100)
            X_train, y_train = smote_n.fit_resample(X_train, y_train)
        else:
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

# Training on train data
if params.MODEL_TYPE == 'FeedForward':

    if params.ORDINAL_LOSS:
        loss_func = lambda x, y: nn.MSELoss(reduction='none')(x, y).sum(axis=1).mean(axis=0)
    else:
        loss_func = params.LOSS_FUNC

    if params.CROSS_VALIDATION:
        accuracies, dcas, ampcas, gmpcas, qwks = k_fold_cross_validation(y_train, X_train, params.K_FOLDS, loss_func, params.LEARNING_RATE, params.NUM_EPOCHS, params.HIDDEN_SIZE)

    model = FeedForward(params.HIDDEN_SIZE, params.PMF_LAYER, params.PMF_TYPE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.REGULARIZATION)
    model, train_loss = train(X_train.to_numpy(), y_train.to_numpy(), model, loss_func, optimizer, params.NUM_EPOCHS)
    test_estimation, train_acc, acc, dca, ampca, gmpca, qwk = test(model, torch.from_numpy(X_train.to_numpy()), torch.from_numpy(y_train.to_numpy()), torch.from_numpy(X_test.to_numpy()), torch.from_numpy(y_test.to_numpy()), True)

    torch.save(model, 'models/model.pt')

    print('========================================================')
    if params.CROSS_VALIDATION:
        print('Accuracy: {}, Confidence Interval: ({},{})'.format(acc, min(accuracies), max(accuracies)))
        print('DCA: {}, Confidence Interval: ({},{})'.format(dca, min(dcas), max(dcas)))
        print('AMPCA: {}, Confidence Interval: ({},{})'.format(ampca, min(ampcas), max(ampcas)))
        print('GMPCA: {}, Confidence Interval: ({},{})'.format(gmpca, min(gmpcas), max(gmpcas)))
        print('QWK: {}, Confidence Interval: ({},{})'.format(qwk, min(qwks), max(qwks)))
    else:
        print('Accuracy: {}'.format(acc))
        print('DCA: {}'.format(dca))
        print('AMPCA: {}'.format(ampca))
        print('GMPCA: {}'.format(gmpca))
        print('QWK: {}'.format(qwk))

elif params.MODEL_TYPE == 'LinearRegression':
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    
    y_pred = np.round_(model.predict(X_test))
    acc = accuracy_score(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    print('\nClassification report')
    #target_names = ['No injury', 'Possible injury', 'Non-incapacitating (minor) injury', 'incapacitating (major) injury', 'fatal injury']
    print(confusion_matrix(y_test, y_pred))#, target_names=target_names))

    accuracies, qwks = k_fold_cross_validation_lin_reg(y_train.to_numpy(), X_train.to_numpy(), params.K_FOLDS)

    print('========================================================')
    if params.CROSS_VALIDATION:
        print('Accuracy: {}, Confidence Interval: ({},{})'.format(acc, min(accuracies), max(accuracies)))
        print('QWK: {}, Confidence Interval: ({},{})'.format(qwk, min(qwks), max(qwks)))
    else:
        print(accuracy_score(y_test, y_pred))

elif params.MODEL_TYPE == 'SVC':
    model = SVC()
    model = model.fit(X_train, y_train)

    y_pred = np.round_(model.predict(X_test))

    print(y_pred)
    print(accuracy_score(y_test, y_pred))
