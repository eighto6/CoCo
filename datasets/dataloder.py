from dgl import DGLGraph
import torch.utils.data
from datasets.prepocessing import IA_processing
from datasets.input_data import LoadAnomaly
from datasets.utils import *
import networkx as nx


def loader_anomaly(args):
    # load and preprocess dataset

    data = LoadAnomaly(args.dataset)
    print(f'normal_class is {args.normal_class}')
    labels, train_mask, val_mask, test_mask = IA_processing(data, args.normal_class, args)
    num = labels[labels == 0]
    print(float(len(num) / len(labels)))
    features = torch.FloatTensor(data.features)
    adj = torch.FloatTensor(data.adj.todense())
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    n_nodes = features.shape[0]


    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
    # graph preprocess and calculate normalization factor
    g = data.graph

    idx = np.arange(len(labels))
    adj_ = sp.coo_matrix((np.ones(labels.shape[0]),
                         (idx, idx)),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    g_cnn = nx.from_scipy_sparse_matrix(adj_, create_using=nx.DiGraph())
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g_cnn.remove_edges_from(nx.selfloop_edges(g_cnn))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
        g_cnn.add_edges_from(zip(g.nodes(), g.nodes()))

    if cuda:
        g = DGLGraph(g).to('cuda:{}'.format(args.gpu))
        g_cnn = DGLGraph(g_cnn).to('cuda:{}'.format(args.gpu))
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

        adj = adj.cuda()

    else:
        g = DGLGraph(g)

    n_edges = g.number_of_edges()
    if args.norm:
        g = norm_graph(g,cuda)




    lpe = laplacian_positional_encoding(g, args.pe_dim)
    if cuda:
        lpe = lpe.to('cuda:{}'.format(args.gpu))
    features = torch.cat((features, lpe), dim=1)
    processed_features = re_features(adj, features, args.hops)
    if cuda:
        processed_features = processed_features.to('cuda:{}'.format(args.gpu))
    return {'g': g, 'features':features, 'processed_features': processed_features, 'labels': labels,
            'train_mask': train_mask,
            'val_mask': val_mask, 'test_mask': test_mask, 'input_dim': in_feats, 'n_classes': n_classes,
            'n_edges': n_edges, 'num_node': n_nodes}


def norm_graph(g,cuda):
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    return g

