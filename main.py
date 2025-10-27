
from sklearn.preprocessing import StandardScaler
from src.evaluation import compute_score
from src.Self_HLCP import Self_HLCP
import scipy.io as scio
import numpy as np


if __name__ == '__main__':
    data = scio.loadmat(f'datasets/realworld/seeds.mat')
    x = data['data']
    y = data['labels']

    s_x = StandardScaler()
    x = s_x.fit_transform(x)
    y = y.reshape(y.shape[0])
    cluster_num = len(np.unique(y))
    n, d = x.shape

    labelnew = Self_HLCP(x, cluster_num, k=10, r=0.8, lam=1)
    ari, nmi, acc = compute_score(labelnew, y)
    print(f"ARI={ari:.2f}, NMI={nmi:.2f}, ACC={acc:.2f}")
