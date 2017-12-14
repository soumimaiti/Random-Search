import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn

def train(x_train,y_train):
    logreg = LogisticRegression(class_weight='balanced', C=0.0001)
    logreg.fit(x_train, y_train)
    return logreg

def get_ber(y_pred, y_test):
    ber = (np.mean(y_pred[y_test == 0] != 0 )+np.mean(y_pred[y_test == 1] != 1 ))/2
    return ber

def get_metrics(x_test, y_test, logreg):
    print('Accuracy: ', logreg.score(x_test, y_test))
    y_pred = logreg.predict(x_test)
    print('BER:', get_ber(y_pred, y_test))
    print(sklearn.metrics.classification_report(y_pred, y_test))
    print('Confusion Matrix\n', sklearn.metrics.confusion_matrix(y_pred, y_test))
    
    
def get_cost(x, y, x_test, y_test):
    #Train or test TODO: decide
    #mean = x.mean(axis=0) 
    #std = x.std(axis=0)+1e-6
    #x_n = (x - mean) / std
    #logreg = train(x_n, y)
    logreg = train(x, y)
    
    #x_test_n = (x_test - mean) / std
    #y_pred = logreg.predict(x_test_n)
    y_pred = logreg.predict(x_test)
    
    n = x_test.shape[0]
    #aic = n*np.log(sum((y_pred-y_test)*(y_pred-y_test))/n)+2*(x_test.shape[1]+1)
    #AIC=n∗ln(SSEn)+2k
    ber = get_ber(y_pred, y_test)
    
    return ber


def get_rcost(x_train, y_train, x_test, y_test, ind):
    cost =  get_cost(x_train[:,ind==1],y_train, x_test[:, ind==1], y_test)
    
    #return cost+0.001*np.mean(ind)
    return cost+0.0001*np.sum(ind)
    #return cost

nbr=6

def rand_init(x_train, y_train, n=50000, seed=123456):
    size = x_train.shape[1]
    ind = np.zeros(size)
    a = np.arange(size)
    np.random.shuffle(a)
    ind[a[:n]] = 1
    return ind

def get_nbr(ind, code=1, nbr=2, seed=123456):
    #print("Code", code, "bits", nbr)
    if code == 1:
        #This part looks for random 1/2 position swap
        return rand_swap_nbr(ind, nbr, seed=123456)
    elif code == 2:
        #This part allows grow/shrink
        return rand_any_nbr(ind, nbr, seed=123456)
    elif code == 3:
        #This part allows grow/shrink
        return rand_mutate_nbr(ind, nbr, seed=123456)
    else:
        print('Wrong code')
    return ind



def get_rand_nbr(ind, code=1, nbr=30, bits=2):
    #print("code", code, "nbr",nbr)
    ind_nbr = np.zeros((nbr, ind.shape[0]))
    for i in range(0, nbr):
        ind_nbr[i] = get_nbr(ind, code, nbr=bits)
      
    return ind_nbr

def rand_swap_nbr(ind, nbr=2, seed=123456):
    '''
    ind --> is a 0/1 array. Neighbours of this array would be maximum of any two positions changed
    '''
    ind_pos, = np.where(ind==1)
    ind_neg, = np.where(ind==0)
    
    
    pos = np.random.choice(ind_pos, nbr/2, replace=False)
    ind[pos] = 0
     
    neg = np.random.choice(ind_neg, nbr/2, replace=False)
    ind[neg] = 1
    return ind


def rand_any_nbr(ind, nbr=2, seed=123456):
    '''
    ind --> is a 0/1 array. Neighbours of this array would be maximum of any two positions changed
    '''
    ind_pos, = np.where(ind==1)
    ind_neg, = np.where(ind==0)
        
    ch = np.random.uniform(0,2)
    if round(ch) == 0:
        #change some 1 to 0
        n = round(np.random.uniform(1,nbr))
        ind[np.random.choice(ind_pos, n)] = 0
    elif round(ch) == 1:
        n1 = round(np.random.uniform(1,nbr))
        ind[np.random.choice(ind_pos, n1)] = 0
        n2 = round(np.random.uniform(1,nbr))
        ind[np.random.choice(ind_neg, n2)] = 1
        
    else:
        n = round(np.random.uniform(1,nbr))
        ind[np.random.choice(ind_neg, n)] = 1
    
    return ind

def rand_mutate_nbr(ind, nbr=2, seed=123456):
    '''
    ind --> is a 0/1 array. Neighbours of this array would be maximum of any two positions changed
    '''
        
    D = ind.shape[0]
    
    #new neighbor
    ind1 = np.zeros(D)
    a = np.arange(D)
    np.random.shuffle(a)
    start_n = np.random.randint(D)
    ind1[a[:start_n]] = 1
    
    #Mutate
    n = np.random.randint(2, size=D)
    n_orig, = np.where(n==1)
    n_new, = np.where(n==0)
    
    ind2 = np.zeros(D)
    ind2[n_orig] = ind[n_orig]
    ind2[n_new] = ind1[n_new]
    
    return ind2

def rand_mutate_nbr1(ind, nbr=2, seed=123456):
    '''
    ind --> is a 0/1 array. Neighbours of this array would be maximum of any two positions changed
    '''
        
    D = ind.shape[0]
    a = np.arange(D)
    rand_ind = np.random.choice(a, nbr)
    
    
    #Generate a 0/1 for rand choice
    ind_phi = np.random.randint(2, size=nbr)
    phi_pos, = np.where(ind_phi==1)
    phi_neg, = np.where(ind_phi==0)
    
    #print(phi_pos, phi_neg)
    #print(rand_ind, rand_ind[phi_pos], rand_ind[phi_neg])
    ind[rand_ind[phi_pos]]=1
    ind[rand_ind[phi_neg]]=0
    '''
    j = 0
    for i in rand_ind:
        if ind_phi[j] == 0:
            if ind[i] == 1:
                ind[i] = 0 
                #flip the selection
        else:
            if ind[i] == 0:
                ind[i] = 1 
                #flip the selection
        j=j+1
    '''
    return ind


import os
import errno

def  save_res(save_path='results', file='Temp', ind=[], ber=0.0,c_a=[]):

    try:
        os.makedirs (save_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        

    np.savez_compressed(os.path.join(save_path, file),
                        ind = ind, ber=ber, cost_arr=c_a)        
    