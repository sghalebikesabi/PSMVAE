import numpy as np
import os
import pandas as pd
import random


# increase timeout variable because of costly read_csv
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT']= '10.00'


def MCAR(data, missingness_ratio = 0.2, seed = 0):
    np.random.seed(seed)
    M = pd.DataFrame(np.random.binomial(size=data.shape, n=1, p=missingness_ratio) == True)
    return(M, None) 


def MAR(data, missingness_ratio=0.2, seed = 0):
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n, m = data.shape
    # sample two features that have enough variation
    while True:
        x1, x2 = np.random.choice(range(m), 2, replace=False)
        med_x1 = np.median(data.iloc[:,x1])
        med_x2 = np.median(data.iloc[:,x2])
        missing = (np.array(data.iloc[:,x1] < med_x1) + np.array(data.iloc[:,x2] > med_x2))>0
        mean_missing = np.mean(missing)
        partial_ratio = (missingness_ratio*m/(m-2))/(mean_missing + 1.e-16)
        if mean_missing > 0 and mean_missing < 0.9:
            break
    M_partial = pd.DataFrame(np.random.binomial(size=(np.sum(missing),m-2), n=1, p=missingness_ratio) == True)
    M_partial2 = np.zeros((n,m-2))
    M_partial2[missing, :] = M_partial
    M = np.zeros((n,m), dtype=bool)
    M[:,np.delete(np.arange(m),[x1,x2])] = M_partial2
    M = pd.DataFrame(M)
    return(M, pd.DataFrame(missing))     


def label_dependent_missingness(data, labels, missingness_ratio=0.2, seed = 0):
    np.random.seed(0)
    unique_labels = labels.iloc[:,0].unique()
    random.shuffle(unique_labels)
    n, m = data.shape
    M = np.zeros((n,m), dtype=bool)
    labels = np.array(labels)
    num_block_missing_feature = int(m*missingness_ratio/0.5)
    j = 0
    for i in unique_labels:
        # block missingness
        start = int(j*m/len(unique_labels)) 
        end = start + num_block_missing_feature
        prob_missing = np.random.uniform(size=(num_block_missing_feature, np.sum(labels==i)))
        block = np.concatenate((np.arange(start, np.min([end, m])), np.arange(0, end-m)))
        M[np.reshape(labels==i, (-1)), np.expand_dims(block, axis=1)] = prob_missing<0.5
        j += 1
    M = pd.DataFrame(M)
    return(M, pd.DataFrame(labels))     


def label_dependent_missingness_no_noise(data, labels, missingness_ratio=0.2, seed = 0):
    np.random.seed(0)
    unique_labels = labels.iloc[:,0].unique()
    random.shuffle(unique_labels)
    n, m = data.shape
    M = np.zeros((n,m), dtype=bool)
    num_block_missing_feature = int(m*missingness_ratio)
    j = 0
    for i in unique_labels:
        # block missingness
        start = int(j*m/len(unique_labels)) 
        end = start + num_block_missing_feature
        block = np.concatenate((np.arange(start, np.min([end, m])), np.arange(0, end-data.shape[1])))
        M[(labels==i).values.reshape((-1)), np.expand_dims(block, axis=1)] = int(np.random.uniform()>0.5)
        j += 1
    M = pd.DataFrame(M)
    return(M, labels)     


def MNARsum(data, missingness_ratio = 0.2, k = 5, seed = 0):
    '''MNAR missingness is induced by creating the same missingness pattern when the sum of all variables is
    within a certain quantile.
    '''
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n,m = data.shape
    M = np.zeros((n,m), dtype=bool)
    weight = np.random.normal(size=m)
    r = pd.Series([np.sum(data.iloc[i,:] * weight) for i in range(n)])
    quantile_old = r.quantile(0)
    for l in range(k):
        quantile_new = r.quantile((l+1)/k)
        save = [rr < quantile_new and rr > quantile_old for rr in r]
        start = int(l*m*missingness_ratio)
        end = int((l+1)*m*missingness_ratio)
        M[save,start:end] = 1
        quantile_old = quantile_new
    M = pd.DataFrame(M)
    return(M, None)     


def MNAR1varMCAR(data, missingness_ratio = 0.2, seed = 0):
    '''let one variable be missing with prob missingness_ratio if it exceeds the medianand 
    MCAR for the remaining variables
    '''
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n,m = data.shape
    M = np.random.binomial(1, missingness_ratio, size=(n, m))

    while True:
        var_ind = np.random.choice(m)    
        var_med = data.loc[:, var_ind].median()
        if (sum(data.loc[:, var_ind] > var_med) > n*0.1) and (sum(data.loc[:, var_ind] > var_med) < n*0.9):
            break
    M[:, var_ind] = 0
    M[data.loc[:, var_ind] > var_med, var_ind] = np.random.binomial(1, 0.5, size=sum(data.loc[:, var_ind] > var_med))
    
    miss_set = data.loc[:, var_ind] > var_med

    M = pd.DataFrame(M)
    return(M, (data.loc[:, var_ind] > var_med))     


def MNAR1var(data, missingness_ratio = 0.2, seed = 0):
    '''let one variable be missing with prob missingness_ratio if it exceeds the median and 
    MCAR for the remaining variables
    '''
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n,m = data.shape
    M = np.zeros((n,m), dtype=bool)

    while True:
        var_ind = np.random.choice(m)    
        var_med = data.iloc[:, var_ind].median()
        if (sum(data.iloc[:, var_ind] > var_med) > n*0.1) and (sum(data.iloc[:, var_ind] > var_med) < n*0.9):
            break
    
    M[data.iloc[:, var_ind] > var_med, var_ind] = np.random.binomial(1, missingness_ratio, size=sum(data.iloc[:, var_ind] > var_med))
    # miss_set = data.loc[:, var_ind] > var_med

    M = pd.DataFrame(M)
    return(M, (data.iloc[:, var_ind] > var_med))     


def MNAR2var(data, missingness_ratio = 0.2, seed = 0):
    '''let one variable be missing with prob missingness_ratio if it exceeds the median'''
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n,m = data.shape
    M = np.zeros((n,m), dtype=bool)

    while True:
        var_ind = np.random.choice(m, size=(2), replace=False)
        var_med_0 = data.loc[:, var_ind[0]].median()
        var_med_1 = data.loc[:, var_ind[1]].median()
        if (sum(data.loc[:, var_ind[0]] > var_med_0) > n*0.1) and (sum(data.loc[:, var_ind[0]] > var_med_0) < n*0.9):
            if (sum(data.loc[:, var_ind[1]] > var_med_1) > n*0.1) and (sum(data.loc[:, var_ind[1]] > var_med_1) < n*0.9):
                break

    M[data.loc[:, var_ind[0]] > var_med_0, var_ind[0]] = np.random.binomial(1, missingness_ratio, size=sum(data.loc[:, var_ind[0]] > var_med_0))
    M[data.loc[:, var_ind[1]] > var_med_1, var_ind[1]] = np.random.binomial(1, missingness_ratio, size=sum(data.loc[:, var_ind[1]] > var_med_1))

    M = pd.DataFrame(M)
    return(M, None)     


def logit_missingness(data, missingness_ratio = 0.2, seed = 0):
    np.random.seed(seed)
    data = pd.DataFrame(data)
    n,m = data.shape
    M = np.zeros((n,m), dtype=bool)

    b = np.exp(missingness_ratio)/(np.exp(missingness_ratio)+1)
    W = np.randn(m)

    P = ((data - data.mean(0))*W + b)

    M[data.loc[:, var_ind] > var_med, var_ind] = np.random.binomial(1, missingness_ratio, size=sum(data.loc[:, var_ind] > var_med))

    M = pd.DataFrame(M)
    return(M, (data.loc[:, var_ind] > var_med))     
