
import numpy as np

import pandas as pd



from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.manifold import TSNE




data_train = pd.read_csv('./vector_data/wdbc_569_31.csv')

data_train_np=np.array(data_train)

train_X=data_train_np[:,:-1]
train_y=data_train_np[:,-1]
print(np.unique(train_y))


########################Test data TSNE ###########################
#max_abs_scaler=preprocessing.MaxAbsScaler()
#test_Xn=max_abs_scaler.fit_transform(test_X)
scaler = preprocessing.StandardScaler()
train_Xn = scaler.fit_transform(train_X)

tsne=TSNE(n_components=2,init='pca',random_state=0, perplexity=100)
tsne.fit_transform(train_Xn)
Xtsne=tsne.embedding_


fig, ax = plt.subplots()
markers = ('s', 'x', 'o', '^', 'v','<','d','p','h','+','1','2','3','4')
colors = ('brown', 'blue', 'lightgreen', 'black', 'red','orange','yellow','crimson','orangered','slategray','navy','darkviolet')
cmap = ListedColormap(colors[:len(np.unique(train_y))])
for idx, cl in enumerate(np.unique(train_y)):
   plt.scatter(x=Xtsne[train_y==cl, 0], y=Xtsne[train_y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plt.legend(['class0', 'class1', 'class2','class3','class4','class5','class6','class7','class8','class9','class10','class11'])
plt.title('wdbc')
plt.show()
fig.savefig('wdbc.eps',dpi=600,format='eps')



# ########################Test data ###########################
# lda = LDA(n_components=3) # config.num_label-1)
# test_X = lda.fit_transform(train_X, train_y)
# markers = ('s', 'x', 'o', '^', 'v','<','d','p','h','+','1','2','3','4')
# colors = ('brown', 'blue', 'lightgreen', 'black', 'red','orange','yellow','crimson','orangered','slategray','navy','darkviolet')
# cmap = ListedColormap(colors[:len(np.unique(train_y))])
# #ax = plt.subplot(111, projection='3d')
# #for idx, cl in enumerate(np.unique(test_y)):
# #   ax.scatter(xs=test_X[test_y==cl, 0], ys=test_X[test_y==cl, 1], zs=test_X[test_y==cl, 2],alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
# for idx, cl in enumerate(np.unique(train_y)):
#    #plt.scatter(x=test_X[test_y==cl, 0], y=test_X[test_y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
#    plt.scatter(x=test_X[train_y == cl, 0], y=test_X[train_y == cl, 0], alpha=0.8, label=cl)
# #plt.scatter(x=Wrong_X[:,0], y=Wrong_X[:,1],alpha=0.8, c='yellow',marker='d')
# plt.legend(['class0', 'class1', 'class2','class3','class4','class5','class6','class7','class8','class9','class10','class11'])
# plt.show()



