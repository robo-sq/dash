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
from sklearn.externals.joblib.parallel import Parallel, delayed
import boto3, botocore
import time
import warnings
import multiprocessing
import scipy
from sklearn import preprocessing
warnings.filterwarnings("ignore")

# params
# can choose bayes_synth, linear_synth or log_synth for dat
dat = 'linear'
print(dat)
k = 100
isParallel = False
data_folder = 'data/'

num_cores = multiprocessing.cpu_count()
print('num_cores:')
print(num_cores)

print('Fetching Files')
if dat == 'linear':
    data = np.loadtxt(open(data_folder + "slice_localization_data.csv", "rb"), delimiter=",", skiprows=1)
    y_cat = data[:,-1]
    X = data[:,:-1]
    all_predictors = list(range(X.shape[1]))
elif dat == 'log':
    X = np.load(data_folder + 'synth_log_X.npy')
    y_cat = np.load(data_folder + 'synth_log_y.npy')
    all_predictors = list(range(X.shape[1]))
elif dat == 'bayes':
    X = np.load(data_folder + 'synth_bayes_X.npy')
    all_predictors = list(range(X.shape[1]))
    X = preprocessing.normalize(X, norm='l2', axis=0)
    d = X.shape[0]
    Lambda = np.eye(d)
    sigma = 1

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


def evaluate_oracle(best_k_predictors, a):
    k_plus_1 = list(best_k_predictors + [a])
    if 'bayes' in dat:
        res = obj_fun(k_plus_1)
    else:
        x_train = X[:,k_plus_1]
        res = obj_fun(x_train, y_cat)

    return res


def run_greedy(kToSelect, parallel=True):
    # get starting set of data
    predictors = [([], 0, 0)]

    # loop through predictors and at each step,
    # add one predictor that increases obj function the most
    # and calculate obj function
    for k in range(kToSelect):

        best_k_predictors = predictors[-1][0]
        r2 = []

        predictor_list = list(set(all_predictors) - set(best_k_predictors))
        if not parallel:
            for predictor in predictor_list:
                k_plus_1 = list(best_k_predictors + [predictor])
                
                if 'bayes' in dat:
                    r2.append(obj_fun(k_plus_1))
                else:
                    x_train = X[:,k_plus_1]
                    r2.append(obj_fun(x_train, y_cat))

        else:
            r2 = Parallel(n_jobs=-1)(delayed(evaluate_oracle)(best_k_predictors, a) for a in predictor_list)

        best_k_plus_1 = best_k_predictors + [predictor_list[np.argmax(r2)]]
        
        end = time.time()
        timePassed = end-start

        predictors.append((best_k_plus_1, np.max(r2), timePassed))
        print(np.max(r2))

    return predictors


start = time.time()
predictors = run_greedy(k, isParallel)
print(predictors)
