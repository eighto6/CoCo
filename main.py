import argparse
from dgl.data import register_data_args
import logging
from optim import Corrgraphtrainer
from datasets import dataloader
from networks.init import init_model
import numpy as np
import torch
from dgl import random as dr
import os
import yaml

def main(args):
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic=True
        dr.seed(args.seed)

    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')
    checkpoints_path = f'./checkpoints/{args.dataset}-{args.datamode}-{args.module}-{args.lamda}+bestcheckpoint.pt'
    logging.basicConfig(filename=f"./log/{args.dataset}-{args.datamode}-{args.module}-{args.lamda}.log", filemode="a",
                        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
    logger = logging.getLogger('CoCo')

    data = dataloader.loader_anomaly(args)
    model = init_model(args, data['input_dim'], len(data['features']))

    model = Corrgraphtrainer.train(args, logger, data, model, checkpoints_path)
    return model

SEEDS = [
  4,8,16,32,52
]
def build_args():
    parser = argparse.ArgumentParser(description='CoCo')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default =0,
                        help="gpu")
    parser.add_argument("--correlation", type=str, default='CCA',
                        help="CCA/Partial/pearson/Erdist/CCAPartial/mic/Mutinf")
    parser.add_argument("--lamda", type=float, help="Weight for CCA")
    parser.add_argument("--alpha", type=float, help="Weight for dec")
    parser.add_argument("--gma", type=int, help="Weight for sce")
    parser.add_argument("--seed", type=int, default=52,
                        help="random seed, -1 means dont fix seed")

    parser.add_argument("--module", type=str, default='CoCo',
                        help="CoCo")
   # parser.add_argument("--dataset", type=str, default='cora',help='cora,citeseer,pubmed,Flickr,BlogCatalog,ACM|| books,reddit,weibo,enron,disney')
    parser.add_argument('--n-worker', type=int, default=1,
                        help='number of workers when dataloading')
    parser.add_argument('--hops', type=int, help='number of hops')
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--pe_dim", type=int, help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0, help="normal class")
    parser.add_argument("--n-epochs", type=int, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
                        help="graph normalization (default=False)")
    parser.add_argument("--datamode", type=str, default='AD',
                        help="AD/N")
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--use_cfg", default=True,action="store_true")
    parser.set_defaults(self_loop=True)
    parser.set_defaults(norm=False)

    args = parser.parse_args()
    return args

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def run(Dataset,args):
    for data in Dataset:
        args.dataset = data
        args = load_best_configs(args, "configs_CoCo.yml")
        if args.gpu > 0:
            os.environ['CUDA_VISIBLE_DEVICE'] = '{}'.format(args.gpu)

        print('model: {}'.format(args.module))

        results_dir = 'logs/final/{}'.format(args.module)
        args.outf = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file2print = open('{}/results_{}.log'.format(results_dir, args.module), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, args.module), 'a+')

        import datetime

        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=file2print)
        print(datetime.datetime.now(), file=file2print_detail)
        print(
            "Model\tDataset\tlayers\thops\tembedding\tlamda\tAUC\tAP\tseed\tepoch",
            file=file2print_detail)

        print("Model\tDataset\tcorrelation\tlayers\thops\tembedding\tlamda\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
        print("Model\tDataset\tcorrelation\tlayers\thops\tembedding\tlamda\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch",
              file=file2print)
        file2print.flush()
        file2print_detail.flush()
        AUCs = {}
        APs = {}
        MAX_EPOCHs = {}

        print("[INFO] Dataset={}, Normal Label={}".format(args.dataset, args.normal_class))

        MAX_EPOCHs_seed = {}
        AUCs_seed = {}
        APs_seed = {}
        for seed in SEEDS:
            # np.random.seed(args.seed ** 2)

            args.seed = seed

            args.name = "%s/%s" % (args.module, args.dataset)
            expr_dir = os.path.join(args.outf, args.name, 'train')
            test_dir = os.path.join(args.outf, args.name, 'test')

            if not os.path.isdir(expr_dir):
                os.makedirs(expr_dir)
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)

            args1 = vars(args)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args1.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

            print(args1)

            print("################", args.dataset, "##################")
            print("################  Train  ##################")
            res = main(args)
            auc_test = res[0]
            ap_test = res[1]
            epoch_max_point = res[2]
            AUCs_seed[seed] = auc_test
            APs_seed[seed] = ap_test
            MAX_EPOCHs_seed[seed] = epoch_max_point

            print(
                "{}\t{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}".format(
                    args.module, args.dataset,args.correlation, args.n_layers, args.hops, args.n_hidden, args.lamda,
                    auc_test, ap_test, seed, epoch_max_point), file=file2print_detail)

            file2print_detail.flush()

        # End For

        MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
        AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
        AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
        APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
        APs_seed_std = round(np.std(list(APs_seed.values())), 4)

        print("Dataset: {} \t  AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={}".format(
            args.dataset, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
            MAX_EPOCHs_seed))

        print(
            "{}\t{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                args.module, args.dataset, args.correlation,args.n_layers, args.hops, args.n_hidden, args.lamda,
                AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed_max
            ), file=file2print)
        file2print.flush()

if __name__ == "__main__":
    args = build_args()
    #
    #Dataset =['books','weibo','disney','reddit','Flickr']
    # ['cora','citeseer','ACM', 'BlogCatalog','inj_cora','inj_amazon']#
    Dataset = ['enron']
    #Dataset = ['inj_amazon']
    run(Dataset,args)
