import pandas as pd
import torch
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
import params
from model import train, FeedForward, labels_to_one_hot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import numpy as np
from accuracy import *

######################################################################################################################

# set random seed for REPRODUCIBILITY
torch.manual_seed(params.SEED)
np.random.seed(params.SEED)
random.seed(params.SEED)

######################################################################################################################

# Reading the data

if params.PREPROCESS:
    input_data = pd.read_csv('dataset/preprocessed_data.csv', sep='\t', index_col=0)
else:
    input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t', index_col=0)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.TEST_FRACTION)
train_df = input_data.sample(frac=0.8, random_state=100)
test_df = pd.concat([input_data, train_df, train_df]).drop_duplicates(keep=False)

# upsampling minority classes
if params.UPSAMPLING:
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
X_test, y_test = test_df.drop(columns=['severity']), test_df['severity']

print(X_train.head())

######################################################################################################################

# Training on train data

model = FeedForward(params.HIDDEN_SIZE, params.PMF_LAYER, params.PMF_TYPE)

# Loss functions
loss_func_mse = nn.MSELoss()
loss_func_mae = nn.L1Loss()
loss_func_cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]))

optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
model, train_loss = train(X_train.to_numpy(), y_train.to_numpy(), model, loss_func_cross_entropy, optimizer, params.NUM_EPOCHS)

######################################################################################################################

# Testing on test data

train_result_dist = model.forward(torch.from_numpy(X_train.to_numpy()).float())
test_result_dist = model.forward(torch.from_numpy(X_test.to_numpy()).float())
train_result = torch.softmax(train_result_dist, dim=0)
test_result = torch.softmax(test_result_dist, dim=0)

# Convert to numpy
train_estimation_dist = np.argmax(train_result_dist.detach().numpy(), axis=1)
test_estimation_dist = np.argmax(test_result_dist.detach().numpy(), axis=1)
train_estimation = np.argmax(train_result.detach().numpy(), axis=1)
test_estimation = np.argmax(test_result.detach().numpy(), axis=1)
train_original = y_train.to_numpy()
test_original = y_test.to_numpy()

# Print results

print('\nClassification report')
target_names = ['No injury', 'Possible injury', 'Non-incapacitating (minor) injury', 'incapacitating (major) injury', 'fatal injury']
print(classification_report(test_original, test_estimation, target_names=target_names))

print('\nConfusion matrix')
print(confusion_matrix(test_original, test_estimation))

train_acc = accuracy_score(train_estimation, train_original)
acc = accuracy_score(test_estimation, test_original)
print(f'\nTrain accuracy: {train_acc}')
print(f'\nTest accuracy: {acc}')

dca = sum(test_estimation == test_original) / len(test_original)
proba = test_result_dist.detach().numpy()
qwk = cohen_kappa_score(test_original, test_estimation, weights='quadratic')

## Discrete Classification Accuracy (DCA)
print("DCA: {dca:.4f}".format(dca=dca))

## Arithmetic Mean Probability of Correct Assignment (AMPCA)
print("AMPCA: {ampca:.4f}".format(ampca=AMPCA(proba, y_test)))

## Geometric Mean Probability of Correct Assignment (GMPCA)
print("GMPCA: {gmpca:.4f}".format(gmpca=GMPCA(proba, y_test)))

# Quadratic Weighted Kappa (QWK)
print("QWK: {qwk:.4f}".format(qwk=qwk))