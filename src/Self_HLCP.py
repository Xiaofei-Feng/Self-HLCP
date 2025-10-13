# 调用第三方库
import os
import math
import scipy
from scipy.io import savemat
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import numpy as np
import warnings
import scipy.sparse
from src.evaluation import compute_score
import scipy.spatial.distance as dist

warnings.filterwarnings("ignore")


def get_K_NN_Rho(x, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances /= np.max(distances)
    return np.sum(np.exp(-distances ** 2), axis=1), distances, indices

def similarity_matrix(x, k, gamma=None):
    rbf = pairwise.rbf_kernel(x, gamma)
    knn_graph = kneighbors_graph(x, n_neighbors=k)
    knn_graph = knn_graph.toarray()
    knn_graph = rbf * knn_graph
    W = (knn_graph + knn_graph.T) / 2
    return W

def get_pre_cluster(x, rho, distances, indices, prun=False):
    def pruning(node, target_pre_center):
        node_ind = indices[node]
        if len(np.where(target_pre_center == node)[0]) <= 0:
            return
        for i in np.where(target_pre_center == node)[0]:
            if i in node_ind:
                pruning(i, target_pre_center)
            else:
                target_pre_center[i] = -1
                pruning(i, target_pre_center)

    n = x.shape[0]
    pre_center = np.ones(n, dtype=np.int32) * -1
    sort_rho = np.flipud(np.argsort(rho))

    for i in range(n):
        min_dis = 0
        dis = np.inf
        find = False
        for j, index in enumerate(indices[i]):
            if index == i:
                continue
            if rho[index] > rho[i]:
                if distances[i, j] < dis:
                    find = True
                    min_dis = j
                    dis = distances[i, min_dis]
        if find:
            pre_center[i] = indices[i, min_dis]

    if prun:
        for i in np.where(pre_center == -1)[0]:
            pruning(i, pre_center)
    zero_index = []
    for i in range(n):
        if pre_center[i] == -1:
            zero_index.append(i)
    for i in sort_rho:
        if pre_center[i] == -1 or pre_center[i] in zero_index:
            continue
        pre_center[i] = pre_center[pre_center[i]]
    return pre_center


def hierarchical_clustering_levels(Y):
    # Perform hierarchical clustering
    Z = linkage(Y, method='single')
    # Number of data points
    n = Y.shape[0]
    # Initialize the level matrix
    Lev = np.zeros((n, n))
    # Initialize cluster levels
    levels = np.zeros(2 * n - 1)  # Levels for all clusters, including merged ones
    # Cluster map to track the points in each cluster
    cluster_map = {i: [i] for i in range(n)}
    # Next cluster index
    next_index = n
    # Update levels and Lev matrix after each merge
    for step in Z:
        # Clusters being merged
        cluster1 = int(step[0])
        cluster2 = int(step[1])
        # Max level of the clusters being merged
        max_level = max(levels[cluster1], levels[cluster2])
        # Indices of the members in both clusters
        indices1 = cluster_map[cluster1]
        indices2 = cluster_map[cluster2]
        # Update the Lev matrix
        for i in indices1:
            for j in indices2:
                Lev[i, j] = max_level + 1
                Lev[j, i] = max_level + 1
        # Update levels of the new cluster
        levels[next_index] = max_level + 1
        # Merge the clusters in the map
        cluster_map[next_index] = indices1 + indices2
        # Increment the next cluster index
        next_index += 1
    return Lev

def adjust_affinity(affinity, constraint_matrix, alpha=0.8, beta=0, EPS=1e-10):
    """Adjust the affinity matrix with constraints."""
    adjusted_affinity = np.copy(affinity)
    degree = np.diag(np.sum(affinity, axis=1))
    degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + EPS))
    affinity_norm = degree_norm.dot(affinity).dot(degree_norm)
    temp_value = np.linalg.inv(
        np.eye(affinity.shape[0]) - alpha * affinity_norm)
    final_constraint_matrix = (1 - alpha) ** 2 * temp_value.dot(constraint_matrix).dot(temp_value)
    final_constraint_matrix /= np.max(np.abs(final_constraint_matrix))
    is_positive = final_constraint_matrix > beta
    no_positive = final_constraint_matrix < -beta
    affinity1 = 1 - (1 - final_constraint_matrix * is_positive) * (
            1 - affinity * is_positive)
    affinity2 = (1 + final_constraint_matrix * no_positive) * (
            affinity * no_positive)
    adjusted_affinity = affinity1 + affinity2 + \
                        (1 - (is_positive | no_positive)) * affinity
    return adjusted_affinity

def E2CP(x, cluster_num, k, lam, must_link, cannot_link, alpha=0.6, gamma=None, beta=0.0):
    W = similarity_matrix(x, k)
    semi_link = must_link - (lam * cannot_link)
    A = adjust_affinity(W, semi_link, alpha, beta)
    result = SpectralClustering(
        n_clusters=cluster_num, affinity='precomputed').fit_predict(A)
    return result

def get_link_matrix(Lev, t, pre_center, r):
    n = len(pre_center)
    k = len(t)
    t = {value: index + 1 for index, value in enumerate(t)}
    pre_center = [t[element] - 1 for element in pre_center]
    must_link = np.zeros((n, n))
    cannot_link = np.zeros((n, n))
    Lev_cannot_link = np.full((k, k), 1) * (np.max(Lev) + 1) - Lev
    for i in range(k):
        Lev_cannot_link[i][i] = 0

    for i in range(n):
        for j in range(i+1,n):
            if pre_center[i] == pre_center[j]:

                must_link[i][j] = must_link[j][i] = 1

    for i in range(n):
        for j in range(i+1,n):
            if pre_center[i] != pre_center[j]:
                must_link_weight = r ** Lev[pre_center[i]][pre_center[j]]
                must_link[i][j] = must_link[j][i] = must_link_weight


    for i in range(n):
        for j in range(i+1, n):
            if pre_center[i] == pre_center[j]:
                cannot_link[i][j] = cannot_link[j][i] = 0

    for i in range(n):
        for j in range(i+1, n):
            if pre_center[i] != pre_center[j]:
                    cannot_link_weight = r ** Lev_cannot_link[pre_center[i]][pre_center[j]]
                    cannot_link[i][j] = cannot_link[j][i] = cannot_link_weight

    return must_link, cannot_link


def Self_HLCP(x, cluster_num, k, norm=True, filter_nois=False, nois_den=0.2, r=1, lam=1):
    rho, distances, indices = get_K_NN_Rho(x, k)
    pre_center = get_pre_cluster(x, rho, distances, indices, prun=False)
    pre_center = pre_center.astype(int)
    t = []
    for i in range(len(pre_center)):
        if pre_center[i] == -1:
            t.append(i)
    for i in range(len(pre_center)):
        if pre_center[i] == -1:
            pre_center[i] = i

    nois = set()
    if filter_nois:
        x_2 = x[rho > nois_den]
        result = ldpsc(x_2, cluster_num, k, filter_nois=False)
        y_last = np.ones_like(pre_center) * -1
        y_last[rho > nois_den] = result
        return y_last

    # 计算各指标
    if 1:
        G = nx.Graph()
        for i in t:
            # if i in nois:
            #     continue
            G.add_node(i)
        # SNN and dist
        if 1:
            cluster_nbr = []
            for i in t:
                a = set()
                for j in np.where(pre_center == i)[0]:
                    if j in nois:
                        continue
                    a.update(indices[j])
                cluster_nbr.append(a - nois)

            snn = np.zeros([len(t), len(t)])
            for t_i in range(len(t)):
                for t_j in range(len(t)):
                    if t_i == t_j:
                        snn[t_i, t_j] = 0
                        continue
                    intersection = cluster_nbr[t_i] & cluster_nbr[t_j]
                    snn[t_i, t_j] = len(intersection)
                    if len(intersection) > 0:
                        sum_rho = 0
                        for ind_i, i in enumerate(intersection):
                            sum_rho += rho[i]
                        G.add_edge(t[t_i], t[t_j], snn=len(intersection), dist=np.linalg.norm(
                            x[t[t_i]] - x[t[t_j]], 2), sum_rho=sum_rho)

        # 计算mar
        mar = np.zeros([len(t), len(t)])
        for index_i, i in enumerate(t):
            for index_j, j in enumerate(t):
                if index_i >= index_j:
                    continue
                if snn[index_i, index_j] == 0:
                    mar[index_i, index_j] = np.inf
                else:
                    set_i = np.where(pre_center == i)[0]
                    set_j = np.where(pre_center == j)[0]
                    # x_set,y_set=np.meshgrid(set_i,set_j)
                    # mar_t=np.linalg.norm((x[x_set]-x[y_set]),axis=2).flatten()
                    mar_t = pairwise.euclidean_distances(
                        x[set_i], x[set_j]).flatten()
                    mar_t.sort()
                    mar_k = 4
                    if len(mar_t) < mar_k:
                        mar_k = len(mar_t)
                    for t_i in mar_t[:mar_k]:
                        mar[index_i, index_j] += t_i
                mar[index_j, index_i] = mar[index_i, index_j]

        for u, v in G.edges:
            G.edges[u, v]['mar'] = mar[t.index(u), t.index(v)]

        # gap
        for i in t:
            set_i = np.where(pre_center == i)[0]
            if len(set_i) > 1:
                distMatrix = pairwise.euclidean_distances(x[set_i])
                distMatrix += np.diag(np.ones(len(set_i)) * np.inf)
                min_d_i = np.min(distMatrix, axis=0)
                rho_set_i = rho[set_i]
                G.nodes[i]['gap'] = np.sum(min_d_i * rho_set_i / np.sum(rho_set_i))
            else:
                G.nodes[i]['gap'] = 0

    for u, v in G.edges:
        tar_2 = G.edges[u, v]['mar'] / \
                (G.nodes[u]['gap'] * G.nodes[v]['gap'] + 1e-6) ** 0.5
        tar_3 = G.edges[u, v]['dist']
        tar_5 = G.edges[u, v]['snn']
        G.edges[u, v]['weight'] = tar_5 / (1 + tar_2) / (1 + tar_3)

    sim = np.zeros([len(t), len(t)])
    for u, v in G.edges:
        sim[t.index(u), t.index(v)] = sim[t.index(
            v), t.index(u)] = G.edges[u, v]['weight']

    t_num = np.zeros_like(t)
    for ind_i, i in enumerate(t):
        t_num[ind_i] = len(np.where(pre_center == i)[0])

    x_center = x[np.array(t)]

    A_t = pairwise.rbf_kernel(x_center, gamma=1)
    if np.max(sim) > 0:
        sim_2 = sim / np.max(sim) + np.identity(len(t))
    else:
        sim_2 = 0
    A = A_t + sim_2 * 1000
    A /= np.max(A)
    lev = hierarchical_clustering_levels(1 - A)
    must_link, cannot_link = get_link_matrix(lev, t, pre_center, r)
    label = E2CP(x, cluster_num, k, lam, must_link, cannot_link, alpha=0.6, beta=0.0)

    return label

