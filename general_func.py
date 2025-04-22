import plotly.figure_factory as ff
from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold,StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
import sklearn
import shap
from statsmodels.stats.multitest import multipletests
from glob import glob
# basic models with default parameters
import xgboost as xgb


def GBDT_m(X_train, y_train, X_test=None, y_test=None):
    train_d = xgb.DMatrix(X_train, label=y_train)
    if X_test is not None:
        test_d = xgb.DMatrix(X_test, label=y_test)
        evals = [(train_d, "self"), (test_d, "test")]
    else:
        evals = [(train_d, "self")]
    pos_weight = len(y_train[y_train == 0]) / \
        len(y_train[y_train == 1])
    xgb_params = {"objective": "binary:logistic",
                  "booster": "gbtree",
                  "scale_pos_weight": pos_weight,
                  'nthread':10,
                  "eval_metric": "aucpr", }
    
    clf = xgb.train(xgb_params,
                    train_d,
                    verbose_eval=False,
                    evals=evals,)
    if X_test is None:
        return clf
    preds = clf.predict(test_d)
    accuracy = sklearn.metrics.balanced_accuracy_score(y_test, preds > 0.5)
    auc = metrics.roc_auc_score(y_test, preds)
    AP = metrics.average_precision_score(y_test, preds)
    return preds, preds > 0.5, clf, accuracy, auc, AP

def knn_m(X_train, y_train, X_test=None, y_test=None):
    knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1, metric="cityblock")
    knn.fit(X_train, y_train)
    if X_test is None:
        return knn
    y_pred = knn.predict_proba(X_test)
    y_pred_label = knn.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, knn, accuracy, auc, AP


def linear_m(X_train, y_train, X_test=None, y_test=None):
    model = sklearn.linear_model.RidgeClassifier(alpha=301.09361391756056, max_iter=848,
                                                 tol=2.6054400792170114e-05,)
    model.fit(X_train, y_train)
    if X_test is None:
        return model
    #y_pred = model.decision_function(X_test)
    y_pred_label = model.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    #auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    return None, y_pred_label, model, accuracy, None, None


def LR_m(X_train, y_train, X_test=None, y_test=None):
    model = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    if X_test is None:
        return model
    y_pred = model.predict_proba(X_test)
    y_pred_label = model.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, model, accuracy, auc, AP


def svmlinear_m(X_train, y_train, X_test=None, y_test=None):
    svc_l = svm.LinearSVC(penalty='l2',
                          class_weight="balanced")
    svc_l.fit(X_train, y_train)
    if X_test is None:
        return svc_l
    #y_pred = svc_l.predict_proba(X_test)
    y_pred_label = svc_l.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    #auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    return None, y_pred_label, svc_l, accuracy, None, None

def svmrbf_m(X_train, y_train, X_test=None, y_test=None):
    svc_l = svm.SVC(class_weight="balanced", kernel="rbf", probability=True)
    svc_l.fit(X_train, y_train)
    if X_test is None:
        return svc_l
    y_pred = svc_l.predict_proba(X_test)
    y_pred_label = svc_l.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, svc_l, accuracy, auc, AP


def RF_m(X_train, y_train, X_test=None, y_test=None):
    rforest = RandomForestClassifier(n_estimators=100,
        n_jobs=10,class_weight = 'balanced_subsample'
    )
    rforest.fit(X_train, y_train)
    if X_test is None:
        return rforest
    y_pred = rforest.predict_proba(X_test)
    y_pred_label = rforest.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, rforest, accuracy, auc, AP

def LR_m_sag(X_train, y_train, X_test=None, y_test=None):
    model = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced", n_jobs=-1,solver='sag',max_iter=5000,
    )
    model.fit(X_train, y_train)
    if X_test is None:
        return model
    y_pred = model.predict_proba(X_test)
    y_pred_label = model.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, model, accuracy, auc, AP


def LR_m_ll(X_train, y_train, X_test=None, y_test=None):
    model = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced", n_jobs=-1,solver='liblinear',
    )
    model.fit(X_train, y_train)
    if X_test is None:
        return model
    y_pred = model.predict_proba(X_test)
    y_pred_label = model.predict(X_test)
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred_label)
    auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
    AP = metrics.average_precision_score(y_test, y_pred[:, 1])
    return y_pred, y_pred_label, model, accuracy, auc, AP

def simple_model_evaluation(X, y_bin,nfold=5,repeat=1):
    bal_acc_list = []
    auc_list = []
    for _n in range(repeat):
        kf = StratifiedKFold(n_splits=nfold)
        for train_index, test_index in kf.split(X, y_bin):
            train_X, test_X = X.iloc[train_index, :], X.iloc[test_index, :]
            train_y, test_y = y_bin[train_index], y_bin[test_index]

            RF = RandomForestClassifier(n_jobs=-1)
            RF.fit(train_X, train_y)
            predict_y_proba = rforest.predict_proba(test_X)
            predict_y_bin = RF.predict(test_X)
            bal_acc = metrics.balanced_accuracy_score(
                                test_y, predict_y)
            auc = metrics.roc_auc_score(test_y, predict_y_proba[:, 1])
            auc_list.append(auc)
            bal_acc_list.append(bal_acc)
    return bal_acc_list,auc_list
            
model_def = {#"kmean": knn_m,
             "linear": linear_m,
             "LR": LR_m_ll,
             "RF": RF_m,
    "GBDT":GBDT_m,
#             "SVMrbf": svmrbf_m,
              "SVMlinear": svmlinear_m
            }

# SHAP value retrieved/parsed



