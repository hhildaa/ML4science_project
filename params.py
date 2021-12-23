import torch
import torch.nn as nn
### RANDOM SEED
SEED = 100

### PREPROCESS DATA YES/NO
PREPROCESS = True
BIN_AGES = False
UPSAMPLING = False
# Upsampling type - random or SMOTE
UPSAMPLING_TYPE = "random" 

### SIZE OF TRAIN-TEST-SPLIT
TEST_FRACTION = .2

### NUMBER OF FOLDS FOR CROSS-VALIDATION
CROSS_VALIDATION = False
K_FOLDS = 10

## MODEL ARCHITECTURE
MODEL_TYPE = 'FeedForward' # ('LinearRegression', 'FeedForward', 'SVC)
INPUT_SIZE = 33
if PREPROCESS:
    INPUT_SIZE += 1
if BIN_AGES:
    INPUT_SIZE += 6
    
HIDDEN_SIZE = 300
OUTPUT_SIZE = 5
PMF_LAYER = False
PMF_TYPE = 'Poisson' # One of 'Poisson', 'Bernoulli'

## TRAINING HYPERPARAMS
LEARNING_RATE = 5e-3
NUM_EPOCHS = 150
REGULARIZATION = 0 # 0 == no regularization

## LOSS FUNCTIONS
ORDINAL_LOSS = False
loss_func_mse = nn.MSELoss()
loss_func_mae = nn.L1Loss()
loss_func_cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1]))
LOSS_FUNC = loss_func_mse

## SAVE NET
SAVE_MODEL = False
SAVE_PATH = 'models/model.pt'