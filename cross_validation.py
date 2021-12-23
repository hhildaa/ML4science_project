import numpy as np
from model import *
from evaluation import *

def build_k_indices(y, k_fold):
    """return indices for k_fold cross validation"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)



def k_fold_cross_validation(y, X, k_fold, loss_func, learning_rate, num_epoch, model_hidden_size):
    """
    do k_fold cross validation of neural net on X and y, using loss_func, learning_rate
    and training each for num_epoch.

    returns list of accuracies, DCAs, AMPCAs, GMPCAs, QWKs 
    """
    X = X.to_numpy()
    y = y.to_numpy()

    # split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # define lists to store the accuracies of training data and test data
    cur_acc_train = []
    cur_acc_test = []
    dcas = []
    ampcas = []
    gmpcas = []
    qwks = []

    k_num = 0

    for k in range(k_fold):
        model = FeedForward(model_hidden_size, params.PMF_LAYER, params.PMF_TYPE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        k_num = k_num+1

        print(f'\nCROSS-VALIDATION: {k_num}. FOLD\n')

        X_test = torch.tensor(X[k_indices[k]])
        y_test = torch.tensor(y[k_indices[k]])
        X_train = torch.tensor(np.vstack([X[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k]))
        y_train = torch.tensor(np.hstack([y[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k]))

        model, _ = train(X_train.numpy(), y_train.numpy(), model, loss_func, optimizer, num_epoch)

        test_estimation, train_acc, acc, dca, ampca, gmpca, qwk = test(model, X_train, y_train, X_test, y_test, False)

        cur_acc_train.append(train_acc)
        cur_acc_test.append(acc)
        dcas.append(dca)
        ampcas.append(ampca)
        gmpcas.append(gmpca)
        qwks.append(qwk)
   
    return cur_acc_test, dcas, ampcas, gmpcas, qwks


def k_fold_cross_validation_lin_reg(y, X, k_fold):
    """
    do k_fold cross validation of Linear Regression on X and y

    returns list of accuracies, QWKs 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # define lists to store the accuracies of training data and test data
    cur_acc_test = []
    qwks = []

    k_num = 0

    for k in range(k_fold):
        model = LinearRegression()


        print(f'\nCROSS-VALIDATION: {k}. FOLD\n')

        X_test = X[k_indices[k]]
        y_test = y[k_indices[k]]
        X_train = np.vstack([X[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])
        y_train = np.hstack([y[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])

        model = model.fit(X_train, y_train)
        y_pred = np.round_(model.predict(X_test))
        print(y_pred)
        
        acc = accuracy_score(y_pred, y_test)
        qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        cur_acc_test.append(acc)
        qwks.append(qwk)

    return cur_acc_test, qwks