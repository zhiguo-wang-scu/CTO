import numpy as np
import pandas as pd
import sklearn.svm as svm

from sklearn.metrics import confusion_matrix

class TSVM(object):
    def __init__(self):
        pass

    def initial(self, kernel='rbf'):
        '''
        Initial TSVM

        Parameters
        ----------
        kernel: kernel of svm
        '''
        self.Cl, self.Cu = 1, 0.001
        self.kernel = kernel
        self.clf = svm.SVC(C=1, kernel=self.kernel, probability=True, gamma='auto')

    def train(self, X1, Y1, X2):
        '''
        Train TSVM by X1, Y1, X2

        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        '''
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y1 = np.expand_dims(Y1, 1)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])
        Y3 = Y3.ravel()

        while self.Cu < self.Cl:
            #            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:

                Y2_d = self.clf.decision_function(X2)  # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d  # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]

                if len(positive_set) and len(negative_set):
                    positive_max_id = positive_id[np.argmax(positive_set)]
                    negative_max_id = negative_id[np.argmax(negative_set)]
                    a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                else:
                    a, b = 0, 0

                if a > 0 and b > 0 and a + b > 2:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    Y3 = Y3.ravel()
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu
            Y2_d = self.clf.decision_function(X1)  # linear: w^Tx + b
            Y1 = Y1.reshape(-1)
            epsilon = 1 - Y1 * Y2_d  # calculate function margin
            epsilon_part=epsilon[epsilon>0]
            epsilon_sum=np.sum(epsilon_part)
            epsilon_avr=epsilon_sum/X1.shape[0]

            Y3_d = self.clf.decision_function(X3)  # linear: w^Tx + b
            Y3 = Y3.reshape(-1)
            epsilon2 = 1 - Y3 * Y3_d  # calculate function margin
            epsilon_part2 = epsilon2[epsilon2 > 0]
            epsilon_sum2 = np.sum(epsilon_part2)
            epsilon_avr2 = epsilon_sum2 / X3.shape[0]



        return (a, b, epsilon_avr,epsilon_avr2)

    def score(self, X, Y):
        '''
        Calculate accuracy of TSVM by X, Y

        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples

        Returns
        -------
        Accuracy of TSVM
                float
        '''
        return self.clf.score(X, Y)

    def predict(self, X):
        '''
        Feed X and predict Y by TSVM

        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features

        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        '''
        return self.clf.predict(X)


    def predict_proba(self, X):
        '''
        Feed X and predict Y by TSVM

        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features

        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        '''
        return self.clf.predict_proba(X)


class TSVM_multilabels(object):
    def fit(self, num_labels, data_train, X_unlabel):
        #         self.Y_len = len(Y)
        self.num_labels = num_labels
        predict = [TSVM() for j in range(0, self.num_labels)]
        # xm = X.shape[0]
        # xn = X.shape[1]
        # XT = np.zeros((self.num_labels, xm, xn))

        xm1 = data_train.shape[0]
        xn1 = data_train.shape[1]
        xn1 = xn1 - 1
        X1T = np.zeros((self.num_labels, xm1, xn1))

        X2T = np.zeros((self.num_labels, X_unlabel.shape[0], X_unlabel.shape[1]))

        A = [[] for _ in range(num_labels)]
        B = [[] for _ in range(num_labels)]
        Ep = [[] for _ in range(num_labels)]
        Ep2 = [[] for _ in range(num_labels)]
        Eps = np.zeros((X_unlabel.shape[0],self.num_labels))
        X1_t = data_train[:,:-1]

        for i in range(self.num_labels):
            # data_train_t = data_train.copy()
            # for j in range(0, len(data_train_t)):
            #     if (data_train_t[j, -1] == i):
            #         data_train_t[j, -1] = 1
            #     else:
            #         data_train_t[j, -1] = -1
            #
            # X1_t = data_train_t[:, :-1]
            Y1_t = 1.0 * (data_train[:,-1] == i) - (data_train[:,-1]  != i)
            #Y1_t = data_train_t[:, -1]

            X2 = X_unlabel

            #scaler = StandardScaler()
            # X1_t = scaler.fit_transform(X1_t)
            # X2_t = scaler.transform(X2)
            # X_t = scaler.transform(X)

            model = TSVM()
            model.initial()
            #         model.train(X1_t, Y1_t, X2_t)
            a, b,ep1,ep2 = model.train(X1_t, Y1_t, X2)
            X1T[i] = X1_t
            X2T[i] = X2
            predict[i] = model


            A[i].append(a)
            B[i].append(b)
            Ep[i].append(ep1)
            Ep2[i].append(ep2)

        #         Y_t=Y.copy()
        #         for j in range(0, len(Y_t)):
        #             if (Y_t[j] == i):
        #                 Y_t[j] = 1
        #             else: Y_t[j] = -1
        #         compute_acc=predict[i].score(X_t,Y_t)
        #         print("the compute_acc is", compute_acc)
        #         print(A,B)

        self.svm = predict
        Ep2_max = np.max(Ep2)
        Ep_max = np.max(Ep)
        return (predict, A, B, Ep_max, Ep2_max,Eps)

    def measure_error(self,num_labels, X, Y, svm):
        #    Y = Y.ravel()
        #         self.Y_len = len(Y)
        Y_predict = [[None] * len(Y)] * num_labels
        for j in range(num_labels):
            Y_predict[j] = svm[j].predict_proba(X)

        predicted_labels = [] * len(Y)

        for j in range(len(Y)):
            listC = []
            listD = []
            for i in range(num_labels):
                listC.append(Y_predict[i][j])
                listD.append(listC[i][1])
            predicted_labels.append(np.argmax(listD))

        correct = np.sum(predicted_labels == Y)
        #     print(str(correct)+" out of "+str(len(predicted_labels))+ " predictions are correct")
        model_acc = correct / len(predicted_labels)
        conf_mat = confusion_matrix(Y, predicted_labels)
        #print('TSVM Trained Classifier Accuracy for One VS rest: ', model_acc)
        #     print('\nConfusion Matrix for One VS rest: \n',conf_mat)
        return (model_acc)
    def predict_label(self,num_labels, X, svm):
        #    Y = Y.ravel()
        #         self.Y_len = len(Y)
        Y_predict = [[None] * X.shape[0]] * num_labels
        for j in range(num_labels):
            Y_predict[j] = svm[j].predict_proba(X)

        predicted_labels = [] * X.shape[0]

        for j in range(X.shape[0]):
            listC = []
            listD = []
            for i in range(num_labels):
                listC.append(Y_predict[i][j])
                listD.append(listC[i][1])
            predicted_labels.append(np.argmax(listD))


        return predicted_labels
    def predict_proba(self, X):
        X_len = X.shape[0]
        Y_predict = [[None] * X_len] * self.num_labels
        for j in range(self.num_labels):
            Y_predict[j] = self.svm[j].predict_proba(X)

        predicted_pro = [] * X_len

        for j in range(X_len):
            listC = []
            listD = []
            listE = []
            for i in range(self.num_labels):
                listC.append(Y_predict[i][j])
                listD.append(listC[i][1])
            arrayD=np.asarray(listD)
            sumD=np.sum(arrayD)
            arrayDn=arrayD/sumD
            predicted_pro.append(arrayDn)

        return predicted_pro






