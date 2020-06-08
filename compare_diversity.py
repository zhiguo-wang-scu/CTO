
from TSVM import *




from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import tree

import time

from sklearn.model_selection import KFold,StratifiedKFold

import pandas as pd
import numpy as np
import sklearn.svm as svm

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

def __get_coefficients(y_true, y_pred_a, y_pred_b):
    a, b, c, d = 0, 0, 0, 0
    for i in range(y_true.shape[0]):
        if y_pred_a[i] == y_true[i] and y_pred_b[i] == y_true[i]:
            a = a + 1
        elif y_pred_a[i] != y_true[i] and y_pred_b[i] == y_true[i]:
            b = b + 1
        elif y_pred_a[i] == y_true[i] and y_pred_b[i] != y_true[i]:
            c = c + 1
        else:
            d = d + 1

    return a, b, c, d
def q_statistics(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    q = float(a * d - b * c) / (a * d + b * c)
    return q
def correlation_coefficient_p(y_true, y_pred_a, y_pred_b):
    a, b, c, d = __get_coefficients(y_true, y_pred_a, y_pred_b)
    e = float((a * d - b * c))
    f =  np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if e ==0 and f ==0:
        p=0
    else:
        p = float((a * d - b * c)) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

    return p
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


####### Co-xgboost and TSVM
acc_train_CXT = pd.DataFrame()
acc_test_CXT = pd.DataFrame()
time_CXT = pd.DataFrame()

####### TSVM
acc_train_TSVM = pd.DataFrame()
acc_test_TSVM = pd.DataFrame()
time_TSVM = pd.DataFrame()
Ep_TSVM = pd.DataFrame()
Ep2_TSVM = pd.DataFrame()
####### co-forest
acc_train_cf = pd.DataFrame()
acc_test_cf = pd.DataFrame()
time_cf = pd.DataFrame()
####### svm
acc_train_svc = pd.DataFrame()
acc_test_svc= pd.DataFrame()
time_svc = pd.DataFrame()
####### xgboost
acc_train_xgboost = pd.DataFrame()
acc_test_xgboost= pd.DataFrame()
time_xgboost = pd.DataFrame()
####### ladder_network
acc_train_ladder = pd.DataFrame()
acc_test_ladder = pd.DataFrame()
time_ladder = pd.DataFrame()

acc_train_cto = pd.DataFrame()
acc_test_cto = pd.DataFrame()
time_cto = pd.DataFrame()

acc_train_cto_ave = pd.DataFrame()
acc_test_cto_ave = pd.DataFrame()
time_cto_ave = pd.DataFrame()

acc_train_cto_no_prior = pd.DataFrame()
acc_test_cto_no_prior = pd.DataFrame()
time_cto_no_prior  = pd.DataFrame()
####### Graph-SVM
acc_train_G = pd.DataFrame()
acc_test_G = pd.DataFrame()
time_G = pd.DataFrame()

####### GMM
acc_train_GMM = pd.DataFrame()
acc_test_GMM = pd.DataFrame()
time_GMM = pd.DataFrame()
tic = time.time()
kf = StratifiedKFold(n_splits=5,shuffle=False)
tree_acc=[]
TSVM_acc=[]
xgboost_acc=[]

tree_tree_diver=[]
tree_TSVM_diver=[]
xgboost_TSVM_diver=[]
for data_num in range(len(s_data)):
    #     data = pd.read_csv(os.path.join(path,s_dataname[data_num]))


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
    tree_acc_local = []
    TSVM_acc_local = []
    xgboost_acc_local = []

    tree_tree_diver_local = []
    tree_TSVM_diver_local = []
    xgboost_TSVM_diver_local = []
    for train_index, test_label_idx in kf.split(X, Y):

        X_train_all = get_value(X, train_index)
        Y_train_all = np.array([i[0] for i in list(get_value(Y, train_index))])
        X_train, X_unlabel, Y_train, Y_unlabel = train_test_split(X_train_all, Y_train_all, test_size=0.80, stratify=Y_train_all)

        X_test = get_value(X, test_label_idx)
       # print(Y_train)

        Y_test = np.array([i[0] for i in list(get_value(Y, test_label_idx))])

        '''
         build the classifiers
        '''
        ### tree
        clf_tree = tree.DecisionTreeClassifier()
        clf_tree = clf_tree.fit(X_train, Y_train)
        pred_unlabel_tree = clf_tree.predict(X_test)
        acc_tree = accuracy_score(Y_test, pred_unlabel_tree)
        print("acc_tree:",acc_tree)
        tree_acc_local.append(acc_tree)

        clf_tree1 = tree.DecisionTreeClassifier()
        clf_tree1 = clf_tree1.fit(X_train, Y_train)
        pred_unlabel_tree1 = clf_tree1.predict(X_test)
        acc_tree1 = accuracy_score(Y_test, pred_unlabel_tree1)
        print("acc_tree1:",acc_tree1)
        ### SVM
        clf_svm = svm.SVC(kernel='rbf')
        clf_svm = clf_svm.fit(X_train, Y_train)
        pred_unlabel_svm = clf_svm.predict(X_test)
        acc_svm = accuracy_score(Y_test, pred_unlabel_svm)
        print("acc_svm:",acc_svm)


        Y1 = np.expand_dims(Y_train, 1)
        data_train_1 = np.concatenate([X_train, Y1], axis=1)
        #         Yl = np.squeeze(Yl)

        A = []
        B = []
        Ep_unlabel = []
        Ep_all = []

        model = TSVM_multilabels()

        one_vs_rest_models, A, B, Ep, Ep2, Eps = model.fit(num_labels, data_train_1, X_unlabel)

        pred_test_tsvm = model.predict_label(num_labels, X_test, one_vs_rest_models)
        test_acc_tsvm = model.measure_error(num_labels, X_test, Y_test, one_vs_rest_models)

        print('TSVM test Classifier Accuracy: ', test_acc_tsvm)
        TSVM_acc_local.append(test_acc_tsvm)


        xgboost_param = {
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
            # 'n_estimators': 100  # default = 500
        }
        xgboost_model = XGBClassifier(**xgboost_param)

        xgboost_model.fit(X_train, Y_train)

        preds_test_xgboost = xgboost_model.predict(X_test)
        acc_test_xgboost = accuracy_score(Y_test, preds_test_xgboost)
        print("acc_test_xgboost:", acc_test_xgboost)
        xgboost_acc_local.append(acc_test_xgboost)



        a,b,c,d = __get_coefficients(Y_test, pred_test_tsvm, pred_unlabel_tree)
        print("a:", a, "b:", b, "c:", c, "d:", d)
        diver_sity = correlation_coefficient_p(Y_test,pred_test_tsvm,pred_unlabel_tree)
        diver_sity1 = correlation_coefficient_p(Y_test,pred_unlabel_tree,pred_unlabel_tree1)
        diver_sity2 = correlation_coefficient_p(Y_test, pred_test_tsvm, preds_test_xgboost)

        tree_tree_diver_local.append(diver_sity1)
        tree_TSVM_diver_local.append(diver_sity)
        xgboost_TSVM_diver_local.append(diver_sity2)


        print("diver_sity:",diver_sity)
        print("diver_sity1:", diver_sity1)
        print("diver_sity2:", diver_sity2)
        print("\n")

    tree_acc.append(np.sum(tree_acc_local)/len(tree_acc_local))
    TSVM_acc.append(np.sum(TSVM_acc_local)/len(TSVM_acc_local))
    xgboost_acc.append(np.sum(xgboost_acc_local)/len(xgboost_acc_local))

    tree_tree_diver.append(np.sum(tree_tree_diver_local)/len(tree_tree_diver_local))
    tree_TSVM_diver.append(np.sum(tree_TSVM_diver_local)/len(tree_TSVM_diver_local))
    xgboost_TSVM_diver.append(np.sum(xgboost_TSVM_diver_local)/len(xgboost_TSVM_diver_local))

np.save("./save/tree_acc.npy",tree_acc)
np.save("./save/TSVM_acc.npy",TSVM_acc)
np.save("./save/xgboost_acc.npy",xgboost_acc)
np.save("./save/tree_tree_diver.npy",tree_tree_diver)
np.save("./save/tree_TSVM_diver.npy",tree_TSVM_diver)
np.save("./save/xgboost_TSVM_diver.npy",xgboost_TSVM_diver)