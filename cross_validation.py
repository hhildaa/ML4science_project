import numpy as np
from model import *
from accuracy import *

def build_k_indices(y, k_fold):

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)



def k_fold_cross_validation(y, X, k_fold, model, loss_func, optimizer, num_epoch):

    X = X.to_numpy()
    y = y.to_numpy()

    # split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # define lists to store the accuracies of training data and test data
    accs_train = []
    accs_test = []

    for i in range(1): # here should be some parameter

        cur_acc_train = np.zeros(k_fold)
        cur_acc_test = np.zeros(k_fold)
        
        cur_pred = np.zeros(k_fold)

        k_num = 0

        for k in range(k_fold):
            k_num = k_num+1

            print(f'\nCROSS-VALIDATION: {k_num}. FOLD\n')

            X_test = torch.tensor(X[k_indices[k]])
            y_test = torch.tensor(y[k_indices[k]])
            X_train = torch.tensor(np.vstack([X[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k]))
            y_train = torch.tensor(np.hstack([y[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k]))

            model, _ = train(X_train.numpy(), y_train.numpy(), model, loss_func, optimizer, num_epoch)

            test_estimation, train_acc, acc, dca, ampca, gmpca, qwk = test(model, X_train, y_train, X_test, y_test, False)

            cur_acc_train[k] = train_acc
            cur_acc_test[k] = acc

        accs_train.append(cur_acc_train.mean())
        accs_test.append(cur_acc_test.mean())

    best_accuracy = np.max(accs_test)

    # todo find best prediction and best parameters
   
    return best_accuracy