from collections import namedtuple
from networkx import read_edgelist, set_node_attributes
from pandas import read_csv, Series
from numpy import array
import random
import numpy as np
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

seed = 2020
random.seed(seed)
np.random.seed(seed)

DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'network']
)


def load_karate_club():
    network = read_edgelist(
        'karate.edgelist',
        nodetype=int)  # 从边列表中读取图形。

    attributes = read_csv(
        'karate.attributes.csv',
        index_col=['node'])  # 以node这一列作为索引

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )

    X_train, y_train = map(array, zip(*[
        ([node], int(data['role'] == 'Administrator'))
        for node, data in network.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ]))
    X_test, y_test = map(array, zip(*[
        ([node], int(data['community'] == 'Administrator'))
        for node, data in network.nodes(data=True)
        if data['role'] == 'Member'
    ]))

    return DataSet(
        X_train, y_train,
        X_test, y_test,
        network)


class GcnLayer(nn.Module):
    def __init__(self, A, in_units, out_units, activation='relu'):
        super(GcnLayer, self).__init__()
        I = np.eye(*A.shape)
        A_hat = A.copy() + I

        D = np.sum(A_hat, axis=0)
        D_inv = D ** -0.5
        D_inv = np.diag(D_inv)

        self.A_hat = (D_inv * A_hat * D_inv).astype('float32')
        self.A_hat = torch.tensor(self.A_hat)

        self._fc = nn.Sequential(nn.Linear(in_units, out_units), nn.ReLU())

    def forward(self, x):
        x = torch.matmul(self.A_hat, x)
        x = self._fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, A, input_dim, hidden_dim_list):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.gcn1 = GcnLayer(A, self.input_dim, hidden_dim_list[0])
        self.gcn2 = GcnLayer(A, hidden_dim_list[0], hidden_dim_list[1])
        self._fc = nn.Sequential(nn.Linear(hidden_dim_list[1], 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    zkc = load_karate_club()
    A = to_numpy_matrix(zkc.network)
    A = np.array(A)
    X_train = zkc.X_train.flatten()
    y_train = zkc.y_train
    X_test = zkc.X_test.flatten()
    y_test = zkc.y_test
    X_1 = np.eye(34, 34).astype('float32')
    X_2 = np.zeros((A.shape[0], 2))
    node_distance_instructor = shortest_path_length(zkc.network, target=33)
    node_distance_administrator = shortest_path_length(zkc.network, target=0)

    for node in zkc.network.nodes():
        X_2[node][0] = node_distance_administrator[node]
        X_2[node][1] = node_distance_instructor[node]

    emb_list = []
    epoch_num = 500
    X = np.hstack([X_1, X_2]).astype('float32')
    my_gcn = GCN(A, X.shape[1], [4, 2])
    optimizer = optim.Adam(my_gcn.parameters())
    for epoch in range(epoch_num):
        for i, train_x in enumerate(X_train):
            train_y = np.array(y_train[i])
            train_y = torch.from_numpy(train_y).reshape(-1)
            var_X = torch.from_numpy(X)
            out = my_gcn(var_X)
            emb=out.clone().detach().numpy()
            out = out[train_x].reshape(1,2)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, train_y.long())
            if epoch % 100 == 0:
                print("epoch: {}, train_id: {}, loss is: {}".format(epoch, i, loss.data))
                emb_list.append(emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    _attributes = read_csv(
        'karate.attributes.csv',
        index_col=['node'])
    output = emb_list[0]
    for i in range(34):
        if _attributes.iloc[i, :]['community'] == 'Administrator':
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='b', alpha=0.5, s=100)
        else:
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='r', alpha=0.5, s=100)
    plt.show()
    output = emb_list[5]
    for i in range(34):
        if _attributes.iloc[i, :]['community'] == 'Administrator':
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='b', alpha=0.5, s=100)
        else:
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='r', alpha=0.5, s=100)
    plt.show()
    output = emb_list[-1]
    for i in range(34):
        if _attributes.iloc[i, :]['community'] == 'Administrator':
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='b', alpha=0.5, s=100)
        else:
            plt.scatter(np.array(output)[i, 0], np.array(output)[i, 1], color='r', alpha=0.5, s=100)
    plt.show()
