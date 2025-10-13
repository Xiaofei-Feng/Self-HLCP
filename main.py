
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from src.evaluation import compute_score
from src.Self_HLCP import Self_HLCP
import scipy.io as scio
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #   MAT
    # data = scio.loadmat(f'datasets/realworld/seeds.mat')
    # x = data['data']
    # y = data['labels']

    #   TXT
    data = np.loadtxt('datasets/synthetic/Jain.txt')

    x = data[:, 1:]
    y = data[:, 0]
    s_x = StandardScaler()
    x = s_x.fit_transform(x)
    # # s_x = MinMaxScaler(feature_range=(0,1))
    # # x = s_x.fit_transform(x)
    y = y.reshape(y.shape[0])
    cluster_num = len(np.unique(y))
    n, d = x.shape

    labelnew = Self_HLCP(x, cluster_num, k=10, r=0.8, lam=1)
    ari, nmi, acc = compute_score(labelnew, y)
    print(f"ARI={ari:.2f}, NMI={nmi:.2f}, ACC={acc:.2f}")

    colors = ['deepskyblue', 'yellowgreen', 'mediumorchid','teal']
    cmap = plt.cm.colors.ListedColormap(colors)

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=labelnew, cmap=cmap, alpha=0.7)

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
