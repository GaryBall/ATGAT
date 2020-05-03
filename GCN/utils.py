from __future__ import print_function

import scipy.sparse as sp  # python中稀疏矩阵相关库
import numpy as np  # python中操作数组的函数
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence  # 稀疏矩阵中查找特征值/特征向量的函数


# 将标签转换为one-hot编码形式
def encode_onehot(labels):
    # set()函数创建一个不重复元素集合
    classes = set(labels)
    # np.identity()函数创建方针，返回主对角线元素为1，其余元素为0的数组
    # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列
    # 同时列出数据和数据下标，一般用在for循环中
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# 加载数据
def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    # str.format()函数用于格式化字符串
    print('Loading {} dataset...'.format(dataset))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # fname：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # (2708,1435)
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 得到data部分 (2708,1433)去掉了第一列和最后一列
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])  # 得到label部分，最后一列
    # build graph
    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 获得索引，论文的id
    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 被引用论文 引用论文
    # 将样本之间的引用关系用样本索引之间的关系表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  # .flatten()的作用是把二维数组压平为一个一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    print(labels.shape[0])
    print(edges)
    print(edges.shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵， np.ones()生成的是全为1的矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵(有向图转无向图)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 打印消息：数据集有多少个节点、多少条边、每个样本有多少维特征
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    # 返回特征的密集矩阵表示、邻接矩阵和标签的one-hot编码
    a = features.todense()
    print(a)
    print(a.shape)
    return features.todense(), adj, labels  # todense的作用是返回一个矩阵（matrix）


def normalize_adj(adj, symmetric=True):
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵
    # D^(-1/2) * A * D^(-1/2)
    if symmetric:
        # A.sum(axis=1)：计算矩阵的每一行元素之和，得到节点的度矩阵D
        # np.power(x, n)：数组元素求n次方，得到D^(-1/2)
        # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        # tocsr()函数将矩阵转化为压缩稀疏行矩阵
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


# 在邻接矩阵中加入自连接
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


if __name__ == "__main__":
    load_data()