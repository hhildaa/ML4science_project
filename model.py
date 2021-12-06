import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import params
import numpy as np
from sklearn.metrics import accuracy_score
from math import factorial, log


def one_hot_to_labels(predictions):
    """
    Convert one-hot decoding into multiclass labels
    Example: 4 classes [0,1,2,3]:
                [1,0,0,0] -> 0
                [0,1,0,0] -> 1 ...

    Input:
            labels -- one-hot encoded class labels
    Returns:
            labels -- numpy array with class labels
    """
    return np.argmax(predictions, axis=1)

def labels_to_one_hot(labels):
    """
    Convert multiclass labels into one-hot decoding
    Example: 4 classes [0,1,2,3]:
                0 -> [1,0,0,0]
                1 -> [0,1,0,0] ...

    Input:
            labels -- numpy array with class labels
    Returns:
            labels -- one-hot encoded class labels
    """
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return one_hot_labels

def train(features, labels, model, lossfunc, optimizer, num_epoch):
    """train a model for num_epoch epochs on the given data
    
    Inputs:
        features: a numpy array
        labels: a numpy array
        model: an instance of nn.Module (or classes with similar signature)
        lossfunc: a function : (prediction outputs, correct outputs) -> loss
        optimizer: an instance of torch.optim.Optimizer
        num_epoch: an int
    
    Returns:
        model: The trained model
        last_loss: The training loss in the last epoch
    """
    # Step 1 - create torch variables corresponding to features and labels
    features = torch.from_numpy(features).float()
    if isinstance(lossfunc,nn.CrossEntropyLoss):
        # cross entropy loss takes class labels instead of one-hot as target labels
        labels = torch.from_numpy(labels)
    else:
        labels = torch.from_numpy(labels_to_one_hot(labels)).float()
    
    for epoch in range(num_epoch):
        # Step 2 - compute model predictions and loss
        y_pred_dist = model.forward(features)
        y_pred = F.softmax(y_pred_dist, dim=0)
        loss = lossfunc(y_pred, labels)
        curr_loss = loss.item()

        # Step 3 - do a backward pass and a gradient update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        if isinstance(lossfunc,nn.CrossEntropyLoss):
            # cross entropy loss takes class labels instead of one-hot as target labels
            acc = accuracy_score(np.argmax(y_pred.detach().numpy(), axis=1), labels.detach().numpy())
        else:
            acc = accuracy_score(np.argmax(y_pred.detach().numpy(), axis=1), np.argmax(labels.detach().numpy(), axis=1))
        
        if epoch % 10 == 0:
            print ('Epoch [%d/%d], Accuracy:%.4f, Loss: %.4f' %(epoch+1, num_epoch, acc, curr_loss))
    
    return model, loss.item()

class PMFLayer(nn.Module):
    def __init__(self, pmf='Poisson', K=5):
        super(PMFLayer, self).__init__()
        self.K = K
        self.index_tensor = torch.FloatTensor([i for i in range(1, K+1)])
        self.pmf = pmf
        if pmf == 'Poisson':
            self.log_index_faculty = torch.FloatTensor([log(factorial(i)) for i in range(1, K+1)])
        elif pmf == 'Bernoulli':
            pass
        else:
            raise ValueError('PMFLayer got invalid pmf: {} (expected one of "Poisson", "Bernoulli")'.format(pmf))

    def forward(self, x):
        y = torch.cat([x for _ in range(self.K)], 1) # copy layer
        if self.pmf == 'Poisson':
            y = (self.index_tensor * torch.log(y))- y - self.log_index_faculty
            return y


class FeedForward(nn.Module):
    def __init__(self, hidden_size, pmf_layer=False, pmf='Poisson'):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(params.INPUT_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.pmf_layer = pmf_layer
        if pmf_layer:
            self.fc3 = nn.Linear(hidden_size, 1)
            self.pmf = PMFLayer(pmf=pmf, K=params.OUTPUT_SIZE)
        else:
            self.fc3 = nn.Linear(hidden_size, params.OUTPUT_SIZE)
            
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)

        if self.pmf_layer:
            y = F.softplus(y)
            y = self.pmf(y)
        else:
            y = torch.sigmoid(y)
        #y = F.softmax(y, dim=0)
        return y