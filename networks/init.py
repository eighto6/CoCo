import torch.nn.functional as F
import torch
from networks.CoCo import TransformerModel,TransformerModel_wodec
def init_model(args,input_dim,input_sdim):

    if args.module in ['CoCo']:
        model = TransformerModel(
                args.hops,
                args.n_hidden,
                input_dim+args.pe_dim,
                args.pe_dim,
                n_layers = args.n_layers,
                hidden_dim = args.n_hidden*2,
                dropout_rate = args.dropout,
        )

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        device = torch.device("cuda:{}".format(args.gpu))
        model.to(device)

    print(f'Parameter number of {args.module} Net is: {count_parameters(model)}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)