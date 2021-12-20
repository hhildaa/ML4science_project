
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

## SIZE (no. of features) OF INPUT AND OUTPUT
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
LEARNING_RATE = 1e-3
NUM_EPOCHS = 250

GRID_SEARCH_PARAMS = {
    'hidden_sizes': [50, 300, 500],
    'learning_rates': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'epochs': [50, 250, 500]
}

# baseline feedforward:
# current best test acc: 0.77
# params: 300, 5e-3, 250(?)

# preprocessed data:
# current best test acc: 0.78
# params: 300, 5e-4, 250(?)
# give weight 2 to 0 and 1 class