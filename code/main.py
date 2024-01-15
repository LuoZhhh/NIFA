import torch
import numpy as np
import argparse
import tqdm

from model import VictimModel
from attack import *
from utils import load_data

parser = argparse.ArgumentParser(description="Fairness Attack Source code")

parser.add_argument('--dataset', default='pokec_z', choices=['pokec_z','pokec_n','dblp'])

parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--T', type=int, default=20, help='sampling times of Bayesian Network')
parser.add_argument('--theta', type=float, default=0.5, help='bernoulli parameter of Bayesian Network')
parser.add_argument('--node', type=int, default=102, help='budget of injected nodes')
parser.add_argument('--edge', type=int, default=50, help='budget of degrees')
parser.add_argument('--alpha', type=float, default=1, help='weight of loss_cf')
parser.add_argument('--beta', type=float, default=1, help='weight of loss_fair')

parser.add_argument('--ratio', type=float, default=0.5, help='node of top ratio uncertainty are attacked')
parser.add_argument('--before', action='store_true')
parser.add_argument('--models', type=str, nargs="*", default=[])
parser.add_argument('--loops', type=int, default=50)

parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='early stop patience')
parser.add_argument('--n_times', type=int, default=5, help='times to run')

parser.add_argument('--device', type=int, default=0, help='device ID for GPU')

args = parser.parse_args()
print(args)

# -----------------------------------main------------------------------------------ 

device = torch.device("cuda", args.device)

if args.before:
    B_ACC = {model:[] for model in args.models}
    B_SP = {model:[] for model in args.models}
    B_EO = {model:[] for model in args.models}
A_ACC = {model:[] for model in args.models}
A_SP = {model:[] for model in args.models}
A_EO = {model:[] for model in args.models}

for i in tqdm.tqdm(range(args.n_times)):
    g, index_split = load_data(args.dataset)
    g = g.to(device)
    in_dim = g.ndata['feature'].shape[1]
    hid_dim = args.hid_dim
    out_dim = max(g.ndata['label']).item() + 1
    label = g.ndata['label']

    if args.before:
        for model in args.models:
            victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
            victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
            acc, sp, eo = victim_model.eval(g, index_split)
            B_ACC[model].append(acc)
            B_SP[model].append(sp)
            B_EO[model].append(eo)
    
    attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
    g = attacker.attack(g, index_split)

    for model in args.models:
        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, name=model)
        victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience)
        acc, sp, eo = victim_model.eval(g, index_split)
        A_ACC[model].append(acc)
        A_SP[model].append(sp)
        A_EO[model].append(eo)

print("================Finished================")
str_format = lambda x, y: "{:.2f}±{:.2f}".format(np.mean(x[y])*100, np.std(x[y])*100)
for model in args.models:
    print("\033[95m{}\033[0m".format(model))
    if args.before:
        print("before acc:{}".format(str_format(B_ACC, model)))
        print("before sp:{}".format(str_format(B_SP, model)))
        print("before eo:{}".format(str_format(B_EO, model)))
    print("after acc:{}".format(str_format(A_ACC, model)))
    print("after sp:{}".format(str_format(A_SP, model)))
    print("after eo:{}".format(str_format(A_EO, model)))

