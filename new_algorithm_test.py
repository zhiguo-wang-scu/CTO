

from Opt_Ensemble import *
from TSVM import *
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

import time

from sklearn.model_selection import KFold,StratifiedKFold

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")






import os

path ='./vector_data/'


s_data = []
s_dataname = []



_legend =["cjs","hill-valley","segment_2310_20","wdbc_569_31","steel-plates-fault",
          "analcatdata_authorship","synthetic_control_6c","vehicle_846_19","German-credit",
          "gina_agnostic_no","madelon_no","texture","gas_drift","dna_no"
           ] #
for i in range(len(_legend)):
    temp = pd.read_csv(os.path.join(path ,_legend[i]+".csv"))
    if temp.shape[0] <= 30000000:
        print(_legend[i])
        s_data.append(temp)
        s_dataname.append(_legend[i]+".csv")



def get_value(X, idxs):
    temp = []
    for i in idxs:
        temp.append(X[i].reshape(1, -1))
    return np.concatenate(temp, axis=0)


#     return temp

def str2num(s, encoder):
    return encoder[s]




kernel = lambda x, y: np.exp(-0.8 * np.linalg.norm(x - y) ** 2)





## Data processing
co_res = {}
co_res["acc"] = []
co_res["time"] = []
others = {}
others['acc'] = []
others['time'] = []
_obs = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
_c = [1, 5, 6, 7, 10, 13, 16]

shapes = []





tic = time.time()
kf = StratifiedKFold(n_splits=5,shuffle=False)
for data_num in range(len(s_data)):



    train_acc_list_cto = []
    test_acc_list_cto = []
    time_list_cto = []


    ## import the data
    data = s_data[data_num]
    print("dataset: " + str(data_num))
    shapes.append(data.shape[0])

    _dic = list(set(data.values[:, -1]))
    num_labels = len(_dic)
    encoder = {}
    for i in range(len(_dic)):
        encoder[_dic[i]] = i

    # shuffle original dataset
    data = data.sample(frac=1)
    X = data.values[:, :-1]
    # X = scale(X)  # scale the X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = np.array([str2num(s, encoder) for s in data.values[:, -1]])

    ### K folder cross validation or random 5 times
    #for kk in range(5):
    for train_index, test_label_idx in kf.split(X, Y):

        X_train_all = get_value(X, train_index)
        Y_train_all = np.array([i[0] for i in list(get_value(Y, train_index))])
        X_train, X_unlabel, Y_train, Y_unlabel = train_test_split(X_train_all, Y_train_all, test_size=0.95, stratify=Y_train_all)

        X_test = get_value(X, test_label_idx)
       # print(Y_train)

        Y_test = np.array([i[0] for i in list(get_value(Y, test_label_idx))])

        '''
         build the classifiers
        '''


        coxgboost_param = {
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'num_class': num_labels,
            'gamma': 0.1,
            'max_depth': 12,
            'lambda': 1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'silent': 1,
            'eta': 0.05,
            'seed': 1000,
            'nthread': 5,
            #'n_estimators': 100  # default = 500
        }


        ### optimal weight
        tic_new = time.time()
        classif_new = {}
        classif_new[0] = XGBClassifier(**coxgboost_param)
        classif_new[1] = XGBClassifier(**coxgboost_param)
        classif_new[2] = XGBClassifier(**coxgboost_param)
        classif_new[3] = TSVM_multilabels()
        classif_new_input = CTO(classif_new, num=4, numClasses=num_labels,case='opt', threshold=0.75) # use the optimal weight
        #classif_new_input = CTO(classif_new, num=4, numClasses=num_labels, case='ave',threshold=0.75)  # use the average weight
        #classif_new_input = CTO(classif_new, num=4, numClasses=num_labels, case='ave',threshold=0.75)  # use the optimal weight without prior
        classif_new_input.fit(num_labels, X_train, Y_train, X_unlabel)
        print("CTO cost time:", time.time() - tic_new)
        time_list_cto.append(time.time() - tic_new)
        ## testing
        pred_cotrain_new = classif_new_input.predict(X_test)
        acc_CTO = accuracy_score(Y_test, pred_cotrain_new)
        ## training
        train_pred_new = classif_new_input.predict(X_train,train=True)
        acc_train_CTO = accuracy_score(Y_train, train_pred_new)

        train_acc_list_cto.append(acc_train_CTO)
        test_acc_list_cto.append(acc_CTO)
        print("CTO train acc is:", acc_train_CTO)
        print("CTO test acc is:", acc_CTO)
        print("\n")




print("\n")