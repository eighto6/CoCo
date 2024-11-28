import scipy.sparse as sp
import scipy.io
import inspect
from torch.utils.data import Dataset
import networkx as nx
import logging
import numpy as np
import torch
logger = logging.getLogger(__name__)

class LoadAnomaly(Dataset):
    def __init__(self,data_source):

        adj, features, labels = self.load_data(data_source)

        labels = np.squeeze(labels)
        adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_ =adj_ + sp.eye(adj_.shape[0])
        self.graph = nx.from_scipy_sparse_matrix(adj_, create_using=nx.DiGraph())
        num_nodes = adj.shape[0]
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        # Some preprocessing
        if data_source in [ 'cora','citeseer','pubmed','BlogCatalog','Flickr','ACM']:
            features_ = sparse_to_tuple(features.tocoo())
            num_features = features_[2][1]
            features_nonzero = features_[1].shape[0]
            features = np.asarray(features.todense())
        else:
            num_features = features.shape[1]
            features_nonzero = num_features
        self.features_nonzero = features_nonzero
        self.adj_label = adj_label
        self.features = features
        self.labels = labels
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_labels = np.max(labels)
        self.adj = adj_

    def load_data(self,data_source):

        if data_source in ["ACM"]:
            data = scipy.io.loadmat("./data/IAdata/{}.mat".format(data_source))
            labels = data["Label"]
            attr_ = data["Attributes"]
            attributes = sp.csr_matrix(attr_)
            network = sp.lil_matrix(data["Network"])

        elif data_source in {'books', 'disney', 'enron', 'reddit', 'weibo',
                             'inj_cora','inj_amazon'}:
            file_path = "./data/IAdata/" + data_source + ".pt"

            data_list = torch.load(file_path)
            attributes = data_list['x'].numpy()
            edge_index = data_list['edge_index'].numpy()
            data = np.ones_like(edge_index[0])
            adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=[attributes.shape[0],attributes.shape[0]])
            network = sp.lil_matrix(adj)
            labels = data_list['y'].numpy()
            labels = labels.astype(int)
        else:
            data = scipy.io.loadmat("./data/IAdata/{}.mat".format(data_source))
            labels = data["Label"]
            attr_ = data["Attributes"]
            attributes = sp.csr_matrix(attr_)
            network = sp.lil_matrix(data["Network"])
        return network, attributes, labels





def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape




