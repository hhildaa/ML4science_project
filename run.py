import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import params
from model import train, FeedForward, labels_to_one_hot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import numpy as np
from accuracy import *

######################################################################################################################

# Reading the data

if params.PREPROCESS:
    input_data = pd.read_csv('dataset/preprocessed_data.csv', sep='\t', index_col=0)
else:
    input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t', index_col=0)

X, y = input_data.drop(columns=['severity']), input_data['severity']
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.TEST_FRACTION)

######################################################################################################################

# Training on train data

model = FeedForward(params.HIDDEN_SIZE)

loss_func_mse = nn.MSELoss()
loss_func_mae = nn.L1Loss()
loss_func_cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]))
optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
model, train_loss = train(X_train.to_numpy(), y_train.to_numpy(), model, loss_func_cross_entropy, optimizer, params.NUM_EPOCHS)

######################################################################################################################

# Testing on test data

test_result = model.forward(torch.from_numpy(X_test.to_numpy()).float())

# Convert to numpy

test_estimation = np.argmax(test_result.detach().numpy(), axis=1)
test_original = y_test.to_numpy()

# Print results

print('\nClassification report')
target_names = ['No injury', 'Possible injury', 'Non-incapacitating (minor) injury', 'incapacitating (major) injury', 'fatal injury']
print(classification_report(test_original, test_estimation, target_names=target_names))

print('\nConfusion matrix')
print(confusion_matrix(test_original, test_estimation))

acc = accuracy_score(test_estimation, test_original)
print(f'\nTest accuracy: {acc}')

dca = sum(test_estimation == test_original) / len(test_original)
proba = test_result.detach().numpy()
qwk = cohen_kappa_score(test_original, test_estimation, weights='quadratic')

## Discrete Classification Accuracy (DCA)
print("DCA: {dca:.4f}".format(dca=dca))

## Arithmetic Mean Probability of Correct Assignment (AMPCA)
print("AMPCA: {ampca:.4f}".format(ampca=AMPCA(proba, y_test)))

## Geometric Mean Probability of Correct Assignment (GMPCA)
print("GMPCA: {gmpca:.4f}".format(gmpca=GMPCA(proba, y_test)))

# Quadratic Weighted Kappa (QWK)
print("QWK: {qwk:.4f}".format(qwk=qwk))