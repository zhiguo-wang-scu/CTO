import numpy as np
# from TSVM_multi1 import *
from TSVM import *
import sklearn
from sklearn.externals.joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from Optimal_weight import *

class CTO:
    def __init__(self, classifier, num, numClasses,case='opt',threshold=0.75):
        self.numClasses = numClasses
        self.num = num
        self.threshold = threshold
        self.classifiers = classifier
        self.n_cores = 12
        self.case = case
    ##  for parallel computing
    def parallel_one(self, classifiers, _sample, num):
        return Parallel(n_jobs=self.n_cores, verbose=False)(
            delayed(classifiers[i].fit)(*_sample[i]) for i in range(num))

    def parallel_two(self, classifiers, L_X, Li_X, L_y, Li_y, update, num):
        return Parallel(n_jobs=self.n_cores, verbose=False)(
            delayed(classifiers[i].fit)(np.append(L_X, Li_X[i], axis=0), np.append(L_y, Li_y[i], axis=0)) for i in
            range(num) if update[i] == True)

    def fit(self, num_labels, L_X, L_y, U_X):
        inbags = [[]] * self.num
        _sample = []
        for i in range(self.num - 1):

            flag = 0
            # _sample = []
            while flag == 0:
                random_seed = np.random.randint(1000)  # 够不够？
                random_state = np.random.RandomState(random_seed)
                inbags[i] = random_state.randint(0, L_X.shape[0],
                                                 size=(L_X.shape[0],))  # mark down the index of sampled data
                sample = sklearn.utils.resample(L_X, L_y, random_state=random_state)  # BootstrapSample(L)
                c_y = len(np.unique(np.array(L_y)))
                c_s = len(np.unique(sample[1]))
                if c_y == c_s:
                    flag += 1

            _sample.append(sample)
            # _sample.append(sample2)
        _temp = self.parallel_one(self.classifiers, _sample, num=self.num - 1)
        for c in range(self.num - 1):
            # self.classifiers[i].fit(sample1,sample2)  # Learn(Si)
            self.classifiers[c] = _temp[c]

        # flag = 0
        # while flag == 0:
        #     random_seed = np.random.randint(1000)  # 够不够？
        #     random_state = np.random.RandomState(random_seed)
        #     inbags[self.num-1] = random_state.randint(0, L_X.shape[0],
        #                                      size=(L_X.shape[0],))  # mark down the index of sampled data
        #     sample1,sample2 = sklearn.utils.resample(L_X, L_y, random_state=random_state)  # BootstrapSample(L)
        #     c_y = len(np.unique(np.array(L_y)))
        #     c_s = len(np.unique(sample2))
        #     if c_y == c_s:
        #         flag += 1
        A = []
        B = []
        if L_X.shape[0] > 15000:
            sample1, X_unlabelll, sample2, Y_unlabelll = train_test_split(L_X, L_y, test_size=0.80, stratify=L_y)
        else:
            sample1, sample2 = L_X, L_y

        sample2 = np.expand_dims(sample2, 1)
        data_train = np.concatenate((sample1, sample2), axis=1)
        # sample2 = np.squeeze(sample2)
        # nn,mm=len(sample1[:,1]),len(sample1[1,:])
        # if nn<1000 or mm < 50:
        nt = len(U_X[:, 1])
        if nt > 20000:
            index11 = np.random.permutation(U_X.shape[0])
            U_XX = U_X[index11]
            len_sub1 = round(U_XX.shape[0] * 0.2)
            U_X1 = U_X[:len_sub1]
        else:
            U_X1 = U_X

        one_vs_rest_models, A, B, Ep, Ep2,Eps = self.classifiers[self.num - 1].fit(num_labels, data_train, U_X1)
        # else:
        #     thresholds = np.argsort(-self.classifiers[self.num - 2].feature_importances_)
        #     sample1_new = sample1[:,thresholds[:30]]
        #     data_train_new = np.concatenate((sample1_new, sample2), axis=1)
        #     one_vs_rest_models, A, B, Ep, Ep2 = self.classifiers[self.num - 1].fit(num_labels, data_train_new, U_X[:,thresholds[:30]])
        # #                 print(A)
        A_min = np.min(A)
        #                 print(A_min)
        B_min = np.min(B)
        d_min = np.min((A_min, B_min))
        self.Ep2 = Ep2
        self.Ep = Ep

        self.d_min = d_min
        if L_X.shape[0] <= 2000:
            self.aa = 0.2
        else:
            self.aa = 0.1
        e_prime = [0.5] * self.num
        l_prime = [0] * self.num
        e = [0] * self.num
        update = [False] * self.num
        Li_X, Li_y = [[]] * self.num, [[]] * self.num  # to save proxy labeled data
        weight = [[]] * self.num
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations



            w, d = self.get_opt_weight(U_X)
            #print(self.weight)

            for i in range(self.num):
                idExclude = i
                update[i] = False
                #                 print("L_X",L_X.shape)
                #                 print("X1T",self.X1T.shape)
                e[i] = self.measure_error(L_X, L_y, idExclude, inbags)  # Estimate Error(H_i,L)
                if e[i] < e_prime[i]:  # if(e_{i,t}<e_{i,t-1})
                    index = np.random.permutation(U_X.shape[0])
                    len_sub = round(U_X.shape[0] * 0.8)
                    U_X2 = U_X[index[:len_sub + 1]]
                    Li_X[i], Li_y[i], weight[i] = self.isHighConfidence_new(U_X2,w, i)

                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if e[i] != 0:
                        numWeight = int(e_prime[i] * l_prime[i] / e[i])  # compute (e_{i,t-1}W_{i,t-1}/e_{i,t})
                    else:
                        numWeight = len(Li_y[i])
                    if len(Li_y[i]) > numWeight:  # W_{i,t}<e_{i,t-1}W_{i,t-1}/e_{i,t}
                        Li_X[i] = Li_X[i][:numWeight]
                        Li_y[i] = Li_y[i][:numWeight]
                        weight[i] = weight[i][:numWeight]
                    sumweight = np.sum(weight[i])
                    if l_prime[i] < sumweight:
                        if e[i] * sumweight < e_prime[i] * l_prime[i]:
                            update[i] = True

            # for i in range(self.num):
            #     if update[i]:
            #         X = np.append(L_X, Li_X[i], axis=0)
            #         Y = np.append(L_y, Li_y[i],axis=0)
            #         ## data augmentation

            #         #X = self.restore(X)
            #         # sm = SMOTE(random_state=42)
            #         # X_res,y_res = sm.fit_sample(X,Y)
            #         if i<=2:
            #             self.classifiers[i].fit(X, Y)  # train the classifier on integrated dataset
            #         else:
            #             data_new = np.concatenate(
            #                 (X, (np.expand_dims(Y, 1))),
            #                 axis=1)
            #             one_vs_rest_models, A, B, Ep,Ep2 = self.classifiers[i].fit(num_labels, data_new, U_X )
            #             self.Ep2 = Ep2
            #             self.Ep = Ep

            #             A_min = np.min(A)
            #             B_min = np.min(B)
            #             d_min = np.min((A_min, B_min))
            #             self.d_min = d_min

            #         # print(np.append(L_y, Li_y[i],axis=0))
            #         e_prime[i] = e[i]
            #         l_prime[i] = np.sum(weight[i])
            temp_cls_0 = [self.classifiers[i] for i in range(self.num - 1)]
            update_cx = [update[i] for i in range(self.num - 1)]
            temp_cls_update = self.parallel_two(temp_cls_0, L_X, Li_X, L_y, Li_y, update_cx, num=self.num - 1)

            flag = 0
            for i in range(self.num):
                if i <= (self.num - 2):

                    if update[i]:
                        # update the classifiers
                        self.classifiers[i] = temp_cls_update[flag]
                        flag += 1
                        e_prime[i] = e[i]
                        l_prime[i] = np.sum(weight[i])
                # else:
                #     if update[i] and len(L_X[:, 1]) <= 1000 and self.Ep > self.aa:
                #         X = np.append(L_X, Li_X[i], axis=0)
                #         Y = np.append(L_y, Li_y[i], axis=0)
                #         data_new = np.concatenate(
                #             (X, (np.expand_dims(Y, 1))),
                #             axis=1)
                #         # if nn < 1000 or mm < 50:
                #         one_vs_rest_models, A, B, Ep, Ep2,eps = self.classifiers[i].fit(num_labels, data_new, U_X)
                #         # else:
                #         #     thresholds = np.argsort(-self.classifiers[self.num - 2].feature_importances_)
                #         #     sample_new = X[:, thresholds[:30]]
                #         #     data_train_new = np.concatenate((sample_new, (np.expand_dims(Y, 1))), axis=1)
                #         #     one_vs_rest_models, A, B, Ep, Ep2 = self.classifiers[i].fit(num_labels, data_train_new, U_X[:,thresholds[:30]])
                #
                #         self.Ep2 = Ep2
                #         self.Ep = Ep
                #
                #         A_min = np.min(A)
                #         B_min = np.min(B)
                #         d_min = np.min((A_min, B_min))
                #         self.d_min = d_min
                #
                #         e_prime[i] = e[i]
                #         l_prime[i] = np.sum(weight[i])

            if self.iter > 3:
                improve = False
            if update == [False] * self.num:
                improve = False  # if no classifier was updated, no improvement

    def restore(self, X):
        u, sigma, v = np.linalg.svd(X)
        m = len(u)
        n = len(v)
        k = len(sigma)
        sigma = sigma + np.random.rand(k)
        b = np.zeros((m, n))
        for j in range(k):
            for i in range(m):
                b[i] += sigma[j] * u[i][j] * v[j]
        return b

    def predict(self, X, train=False):  # predict hard labels
        num_inst = X.shape[0]
        self.train = train
        pred = self.classifyInstance(X)
        return pred

    def score(self, pred, y):  # return accuracy score
        return sklearn.metrics.accuracy_score(y, pred)

    def measure_error(self, X, y, idExclude, inbags):  # measure out-of-bag error
        #         y = np.squeeze(y)
        num_inst = X.shape[0]
        count = 0
        err = 0
        pred_prob = self.OutofBagDistributionForInstanceExcluded(X, idExclude, inbags)
        a = self.getConfidence(pred_prob)
        b = a[a > self.threshold]  # probability
        c = np.argmax(pred_prob, 1)
        d = c[a > self.threshold]  # label
        y_temp = y[a > self.threshold]  # true label of selected data
        count = len(b)
        e = d[d != y_temp]
        err = len(e)
        if count == 0:
            return 0
        else:
            return err / count


    def get_opt_weight(self,inst):
        distr_all = [[]] * self.num
        for i in range(self.num):
            aa = self.classifiers[i].predict_proba(inst)
            distr_all[i] = np.asarray(aa)
            # if i==0:
            #     dist = distr_all[0]
            # else:
            #     dist = np.concatenate((dist,distr_all[i]),axis=1)
        #dis_list=[[]]*len(inst[:,0])

        opt = opt_weight1(distr_all,len(inst[:,0]),self.num, self.numClasses, self.aa, self.Ep)
        weight = opt.get_w(case=self.case)
        self.weight = weight
        # for i in range(len(inst[:,0])):
        #     dis_list[i] = dist[i].reshape(self.num,self.numClasses)
        #     opt = opt_weight(dis_list[i],self.num,self.numClasses,self.aa,self.Ep2)
        #     weight[i] = opt.get_w()
        return  weight,distr_all
    def distribution_new(self, inst,weight,idExclude):

        num_inst = len(inst[:,0])
        res = np.zeros([num_inst, self.numClasses])
        for i in range(self.num):
            if i ==idExclude:
                continue
            else:
                aa = self.classifiers[i].predict_proba(inst)
                res = res + weight[i]*np.asarray(aa)

        sumtemp = np.sum(res, axis=1)
        for i in range(num_inst):
            res[i, :] = res[i, :] / sumtemp[i]
        return res
    # def distribution_new1(self, dis_list):
    #     num_inst = len(dis_list[0][:,0])
    #
    #     res = np.zeros([num_inst, self.numClasses])
    #     for i in range(self.num):
    #
    #         res = res + self.weight[i]*dis_list[i]
    #
    #     sumtemp = np.sum(res, axis=1)
    #     for i in range(num_inst):
    #         res[i, :] = res[i, :] / sumtemp[i]
    #     return res
    def isHighConfidence_new(self,inst, w,idExclude):  # label the unlabeled data whose confidence is larger than the threshold
        #w,d = self.get_opt_weight(inst)
        distr = self.distribution_new(inst,w,idExclude)

        confidence = self.getConfidence(distr)
        Li_X = inst[confidence > self.threshold]
        Li_ytemp = distr[confidence > self.threshold]
        Li_y = np.argmax(Li_ytemp, 1)
        weight = confidence[confidence > self.threshold]
        return Li_X, Li_y, weight





    def OutofBagDistributionForInstanceExcluded(self, inst, idExclude, inbags):
        num_inst = inst.shape[0]
        res = np.zeros([num_inst, self.numClasses])
        for i in range(self.num):
            if i == idExclude:
                continue

            else:

                distr = self.classifiers[i].predict_proba(inst)
                distr = np.asarray(distr)

            distr[inbags[i]] = np.zeros(self.numClasses)
            res = distr + res
        sumtemp = np.sum(res, axis=1)
        for i in range(num_inst):
            if sumtemp[i] == 0:
                res[i, :] = np.zeros(self.numClasses)
            else:
                res[i, :] = res[i, :] / sumtemp[i]
        return res
    def distributionForInstance(self, inst):  # probability distribution
        num_inst = inst.shape[0]
        res = np.zeros([num_inst, self.numClasses])

        for i in range(self.num):

            distr = self.classifiers[i].predict_proba(inst)
            distr = self.weight[i]*np.asarray(distr)


            res = distr + res
        # nn, mm = len(inst[:, 1]), len(inst[1, :])
        # if nn < 1000 or mm < 50:
        # distr3 = self.classifiers[3].predict_proba(inst)
        # distr3 = np.asarray(distr3)

        # if self.train==False and self.Ep2<=self.aa:
        #     res = (1-3*self.Ep2) * distr3 + self.Ep2*res



        sumtemp = np.sum(res, axis=1)
        for i in range(num_inst):
            res[i, :] = res[i, :] / sumtemp[i]
        return res
    # def distributionForInstance(self, inst):  # probability distribution
    #
    #     num_inst = inst.shape[0]
    #     res = np.zeros([num_inst, self.numClasses])
    #     distr_all = [[]] * self.num
    #     for i in range(self.num):
    #         aa = self.classifiers[i].predict_proba(inst)
    #         distr_all[i] = np.asarray(aa)
    #
    #
    #     sumtemp = np.sum(res, axis=1)
    #     for i in range(num_inst):
    #         res[i, :] = res[i, :] / sumtemp[i]
    #     return res

    def getConfidence(self, p):  # choose the largest probability as the confidence
        max = np.max(p, 1)
        return max

    def classifyInstance(self, inst):  # hard label
        distr = self.distributionForInstance(inst)
        return np.argmax(distr, 1)



