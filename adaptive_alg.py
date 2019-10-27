import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from statsmodels.api import OLS
import itertools
import math
import random
import scipy.sparse
from sklearn.externals import joblib
from sklearn.externals.joblib.parallel import Parallel, delayed
import multiprocessing
import boto3, botocore
import time
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")

# params
# can choose bayes_synth, linear_synth or log_synth for dat
dat = 'linear'
print(dat)
data_folder = 'data/'
isParallel = False
ep = 0.05
alpha = 0.95
# (number of elements to select, number of rounds, guess for OPT)
val_list = [(40,2,.75)]

num_cores = multiprocessing.cpu_count()
print('num_cores:')
print(num_cores)

print('Fetching Files')
if dat == 'linear':
    data = np.loadtxt(open(data_folder + "slice_localization_data.csv", "rb"), delimiter=",", skiprows=1)
    y_cat = data[:,-1]
    X1 = data[:,:-1]
    all_predictors = list(range(X1.shape[1]))
elif dat == 'log':
    X1 = np.load(data_folder + 'synth_log_X.npy')
    y_cat = np.load(data_folder + 'synth_log_y.npy')
    all_predictors = list(range(X1.shape[1]))
elif dat == 'bayes':
    X1 = np.load(data_folder + 'synth_bayes_X.npy')
    all_predictors = list(range(X1.shape[1]))
    X = preprocessing.normalize(X1, norm='l2', axis=0)
    d = X1.shape[0]
    Lambda = np.eye(d)
    sigma = 1

print('Finished Parsing Files')

print('Objective function')
if dat == 'linear':
    def obj_fun(x_t, y_t):
        model = OLS(y_t, x_t).fit()
        pred = model.predict(x_t)
        return r2_score(y_t, pred)
elif dat == 'bayes':
    def obj_fun(S):
        trace_sig = np.trace(np.linalg.inv(Lambda))
        X1 = X[:,S]
        return trace_sig - np.trace(np.linalg.inv(Lambda + sigma**(-2)*np.matmul(X1, X1.T)))
elif dat == 'log':
    def log_likelihood(features, target, weights):
        scores = np.dot(features, weights)
        ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
        return ll

    # we use log likelihood to maximize but plot classification rate
    def obj_fun(x_t, y_t):
        logitm = LogisticRegression(C = 1e15,solver='lbfgs')
        logitm.fit (x_t, y_t)
        y_logit = logitm.predict(x_t)
        class_rate = np.sum(y_logit == y_t)/len(y_t)

        return 1/(-1*log_likelihood(x_t, y_t, logitm.coef_[0]))

    def get_class_rate(x_t, y_t):
        # Create logistic regression object
        logitm = LogisticRegression(C = 1e15,solver='lbfgs')
        logitm.fit (x_t, y_t)
        y_logit = logitm.predict(x_t)
        class_rate = np.sum(y_logit == y_t)/len(y_t)

        return class_rate

if 'bayes' in dat:
    def oracle(cols):
        if cols == []:
            return 0.0
        else:

            r2 = obj_fun(cols)
            return r2
else:
    # given set of features, return obj function
    def oracle(cols):
        if cols == []:
            return 0.0
        else:
            r2 = obj_fun(X1[:, cols], y_cat)
            return r2

# get random sample of size num from array X
def randomSample(X, num):
    # if not enough elements, just return X
    X_copy = X.copy()
    if len(X_copy) < int(num):
        R = X_copy
    else:
        R = [X_copy[i] for i in sorted(random.sample(range(len(X_copy)), int(num)))]
    return R

# estimate set value by sampling m times
def estimateSet(X, S, k, r, m=5):
    est = 0
    fS = oracle(S)
    # repeat m times
    for it in range(m):
        # sample size k/r m times
        R = randomSample(X, k/r)
        est += oracle(R + S)
    return (est - m*fS)/m

# estimate marginal contribution by sampling m times
def estimateMarginal(X, S, a, k, r, m=5):
    est = 0
    # repeat m times
    for it in range(m):
        # if there are not enough elements
        R = randomSample(X, k/r)
        marg1 = oracle(R+S+[a])
        if a in R:

            R.remove(a)
            marg2 = oracle(S + R)
        else:
            marg2 = oracle(S + R)      
        est += marg1 - marg2
    return est/m


def union(A, B):
    return list(set(A + B))

# DASH
def dash(k, r, ep, OPT, X, alpha, debug=True, parallel=False):
    
    m = 5
    S = []

    # cache values for plotting
    y_adap = []

    # for r iterations
    for i in range(r):
        T = []
        print(i)

        fS = oracle(S)
        fST = oracle(union(S,T))

        while ((fST - fS) < (ep/20)*(OPT - fS)) and (len(union(S,T)) < k) and len(X)>1:
            # FILTER Step
            # this only changes X
            vs = estimateSet(X, union(S,T), k, r, m)
            while (vs < alpha**(2)*(1-ep)*(OPT - fST)/r) and (len(X) > 0):
                # get marginal contribution
                if parallel:
                    marg_a = Parallel(n_jobs=-1)(delayed(estimateMarginal)(X, union(S,T), a, k, r, m) for a in X)
                else:
                    marg_a = [estimateMarginal(X, union(S,T), a, k, r, m) for a in X]

                Xnew = [X[idx] for idx, el in enumerate(marg_a) if el >= alpha*(1+ep/2)*(1-ep)*(OPT - fST)/k]
                X = Xnew
                vs = estimateSet(X, union(S,T), k, r, m)

            # update sets
            R = randomSample(X, k/r)
            T = union(T, R)
            # T changes but S doesn't
            fST = oracle(union(S,T))

            # outer loop numbers
            print('Outer Loop: Val')
            print(len(union(S,T)))
            print(fST)

        S = union(S,T)
        fS = oracle(S)
        end = time.time()
        timePassed = end-start

        if 'bayes' in dat:
            y_adap.append((len(S),obj_fun(S), timePassed, fS))
        else:
            y_adap.append((len(S), obj_fun(X1[:,S], y_cat), timePassed, fS))
    return y_adap


res_dict = {}
for i in val_list:
    print(i)
    print('k=' + str(i[0]))
    print('OPT=' + str(i[2]))
    start = time.time()
    y_adap = dash(i[0], i[1], ep, i[2], all_predictors, alpha, parallel=isParallel)
    res_dict.update({i[0]: y_adap})

print(res_dict)

