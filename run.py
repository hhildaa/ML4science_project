import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import params
from model import train, FeedForward, labels_to_one_hot

input_data = pd.read_csv('dataset/tempe_cleaneddata.csv', sep='\t')

X, y = input_data.drop(columns=['severity']), input_data['severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.TEST_FRACTION)

model = FeedForward(params.HIDDEN_SIZE)

loss_func_mse = nn.MSELoss()
loss_func_mae = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=params.LEARNING_RATE)
model, train_loss = train(X_train.to_numpy(), y_train.to_numpy(), model, loss_func_mae, optimizer, params.NUM_EPOCHS)