import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

# Definitions for AMPCA and GMPCA

def AMPCA(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + proba[i, sel_mode]
        i += 1
    N = i-1
    if N == 0: return 0
    return sum/N

def CEL(proba, y):
    sum = 0
    i = 0
    for sel_mode in y:
        sum = sum + np.log(proba[i, sel_mode])
        i += 1
    N = i-1
    return -sum/N

def GMPCA(proba, y):
    return np.exp(-CEL(proba, y))

def test(model, X_train, y_train, X_test, y_test, print_results=False):

    # Testing on test data
    train_result = model.forward(torch.from_numpy(X_train.numpy()).float())
    test_result = model.forward(torch.from_numpy(X_test.numpy()).float())
    #train_result = torch.softmax(train_result_dist, dim=1)
    #test_result = torch.softmax(test_result_dist, dim=1)

    # Convert to numpy
    train_estimation_dist = np.argmax(train_result.detach().numpy(), axis=1)
    test_estimation_dist = np.argmax(test_result.detach().numpy(), axis=1)
    train_estimation = np.argmax(train_result.detach().numpy(), axis=1)
    test_estimation = np.argmax(test_result.detach().numpy(), axis=1)
    train_original = y_train.numpy()
    test_original = y_test.numpy()

    # Calculate accuracies
    train_acc = accuracy_score(train_estimation, train_original)
    acc = accuracy_score(test_estimation, test_original)

    dca = sum(test_estimation == test_original) / len(test_original)
    proba = test_result.detach().numpy()
    print(proba)
    qwk = cohen_kappa_score(test_original, test_estimation, weights='quadratic')
    ampca=AMPCA(proba, y_test)
    #gmpca = 0
    gmpca=GMPCA(proba, y_test)

    # Print results
    if print_results:

        print('\nClassification report')
        target_names = ['No injury', 'Possible injury', 'Non-incapacitating (minor) injury', 'incapacitating (major) injury', 'fatal injury']
        print(classification_report(test_original, test_estimation, target_names=target_names))

        print('\nConfusion matrix')
        print(confusion_matrix(test_original, test_estimation, labels=list(range(5))))

        print('Total number of samples: ', len(test_original))
        print('Predicted number of samples: ', len(test_estimation))
        
        print(f'\nTrain accuracy: {train_acc}')
        print(f'\nTest accuracy: {acc}')

        ## Discrete Classification Accuracy (DCA)
        print("DCA: {dca:.4f}".format(dca=dca))

        ## Arithmetic Mean Probability of Correct Assignment (AMPCA)
        print("AMPCA: {ampca:.4f}".format(ampca=ampca))

        ## Geometric Mean Probability of Correct Assignment (GMPCA)
        print("GMPCA: {gmpca:.4f}".format(gmpca=gmpca))

        # Quadratic Weighted Kappa (QWK)
        print("QWK: {qwk:.4f}".format(qwk=qwk))

    return test_estimation, train_acc, acc, dca, ampca, gmpca, qwk