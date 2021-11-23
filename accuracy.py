import numpy as np
import pandas as pd

# standard accuracy function, counts the correct labels
def accuracy(y_pred, labels):
    acc = (labels == y_pred).sum() / len(y_pred)
    return acc

# These functions are from the original approach

# Definitions for AMPCA and GMPCA
def AMPCA(proba, test_set, choice_col):
    sum = 0
    i = 0
    for sel_mode in test_set[choice_col].values:
        sum = sum + proba[i,sel_mode]
        i += 1
    N = i-1
    return sum/N

def CEL(proba, test_set, choice_col):
    sum = 0
    i = 0
    for sel_mode in test_set[choice_col].values:
        sum = sum + np.log(proba[i,sel_mode])
        i += 1
    N = i-1
    return -sum/N

def GMPCA(proba, test_set, choice_col):
    return np.exp(-CEL(proba, test_set, choice_col))

# Quadratic weighted kappa metric
def QWK(confusion_matrix, actual, predicted_vector, pred=None):
    N = len(confusion_matrix)
    print("confusion matrix:")
    print(pd.DataFrame(confusion_matrix).round(2))
    
    w = np.zeros((N, N))
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/16)
    print('weights matrix:')
    print(w)
    
    act_hist=np.zeros([N])
    for item in actual:
        act_hist[item]+=1
    print("Actual histogram")
    print(act_hist)
    
    if pred is None:
        pred_hist = predicted_vector.sum(axis=0)
    print("Predicted histogram")
    print(pred_hist)
    
    E = np.outer(act_hist, pred_hist)
    E = E/E.sum() # normalize E
    print('Expected matrix (normalized):')
    print(pd.DataFrame(E))

    cm_norm = confusion_matrix/confusion_matrix.sum()
    print('Confusion matrix (normalized):')
    print(pd.DataFrame(cm_norm))
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*cm_norm[i][j]
            den+=w[i][j]*E[i][j]

    weighted_kappa = (1 - (num/den))
    print('weighted kappa:')
    return weighted_kappa