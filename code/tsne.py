import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import dgl
from datetime import datetime

from attack import *
from utils import load_data

parser = argparse.ArgumentParser(description="Fairness Attack Source code")

parser.add_argument('--dataset', default='pokec_z', choices=['pokec_z','pokec_n','NBA', 'dblp', 'german', 'bail', 'credit'])
# parser.add_argument('--sens_attr', default='gender', help='sensitive attribute, ["gender","region"] for ["pokec_z","pokec_n"] and ["country"] for "NBA"')
parser.add_argument('--model', default='GCN', help='core of surrogate model', choices=['GCN', 'SGC', 'APPNP', 'GAT', 'GraphSAGE'])

parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--T', type=int, default=20, help='sampling times for Bayesian Network')
parser.add_argument('--theta', type=float, default=0.5, help='bernoulli parameter for Bayesian Network')
parser.add_argument('--node', type=int, default=102, help='budget for injected nodes')
parser.add_argument('--edge', type=int, default=100, help='budget for degrees')
parser.add_argument('--acc', type=float, default=0.7, help='the selected GNN accuracy on val would be at least this high')
parser.add_argument('--alpha', type=float, default=1, help='weight of kl_loss')
parser.add_argument('--beta', type=float, default=1, help='weight of ce_loss')
parser.add_argument('--gamma', type=float, default=1, help='weight of fairness loss')

parser.add_argument('--link_type', type=str, default='base', choices=['base', 'v1', 'v2'])
parser.add_argument('--ratio', type=float, default=0.5, help='top ratio uncertainty nodes are attacked')
parser.add_argument('--ce_ratio', type=float, default=1)
parser.add_argument('--before', action='store_true')
parser.add_argument('--models', type=str, nargs="*", default=[])
parser.add_argument('--loops', type=int, default=50)

parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='early stop patience')
parser.add_argument('--n_times', type=int, default=5, help='times to run')

# parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=int, default=2, help='device ID for GPU')
parser.add_argument('--attack', action='store_true')
parser.add_argument('--embed', action='store_true')

args = parser.parse_args()
# print(args)

# -----------------------------------主函数------------------------------------------

# TODO: 测试的时候先不写日志文件了，之后最终运行的时候改回去
# logging.basicConfig(filename=f'../result/logs/{args.dataset}_{datetime.now().strftime("%Y_%m_%d_%H_%M.log")}',
logging.basicConfig(
                    format='%(asctime)s %(name)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('main  ')
# logger.info(args)

device = torch.device("cuda", args.device)

path_graph = 'output/graph/{}.bin'.format(args.dataset)
path_embed = 'output/embed/{}.bin'.format(args.dataset)

if args.embed:
    if args.attack:
        # 进行攻击
        g, index_split = load_data(args.dataset)
        if args.dataset == "pokec_z":
            g.ndata['sensitive'] = 1 - g.ndata['sensitive']
        g = g.to(device)
        in_dim = g.ndata['feature'].shape[1]
        hid_dim = args.hid_dim
        out_dim = max(g.ndata['label']).item() + 1
        label = g.ndata['label']
        
        logger.info("Start to perform fairness attack.")
        attacker = Attacker(g, in_dim, hid_dim, out_dim, device, args)
        g, variance = attacker.attack(g, index_split)
        g = g.cpu()
        dgl.save_graphs(path_graph, [g])

    else:
        #直接用攻击好的图
        glist, _ = dgl.load_graphs(path_graph)
        g = glist[0]
    
    # TSNE嵌入
    from sklearn.manifold import TSNE
    feature = g.ndata['feature']
    feature_embed = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100, early_exaggeration=12.0).fit_transform(feature)

    g.ndata['embed'] = torch.tensor(feature_embed)
    dgl.save_graphs(path_embed, [g])
else:
    # 直接用现成的结果
    glist, _ = dgl.load_graphs(path_embed)
    g = glist[0]
    feature_embed = g.ndata['embed'].numpy()
# ------------------------------------------------------------

# import pdb; pdb.set_trace()
origin_embed = feature_embed[g.ndata['label']>=0]
origin_embed_0 = feature_embed[torch.logical_and(g.ndata['label']>=0, g.ndata['sensitive']==0)]
origin_embed_1 = feature_embed[torch.logical_and(g.ndata['label']>=0, g.ndata['sensitive']==1)]
fake_embed_0 = feature_embed[-args.node:-args.node+args.node//2]
fake_embed_1 = feature_embed[-args.node+args.node//2:]

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(origin_embed[:,0], origin_embed[:,1], s=0.5, c='pink')
# plt.scatter(origin_embed_0[:,0], origin_embed_0[:,1], s=0.5, c='blue')
# plt.scatter(origin_embed_1[:,0], origin_embed_1[:,1], s=0.5, c='red')
plt.scatter(fake_embed_0[:,0], fake_embed_0[:,1], s=5, c='green')
plt.scatter(fake_embed_1[:,0], fake_embed_1[:,1], s=5, c='green')
plt.savefig('output/tsne-{}.png'.format(args.dataset), dpi=500)
print(f"{args.dataset} Finished.")


# python tsne.py --dataset pokec_z --alpha 0.01 --beta 1 --gamma 4 --node 102 --edge 50 --device 0 --models 'GCN' --embed --attack 

# python tsne.py --dataset pokec_n --alpha 0.01 --beta 1 --gamma 4 --node 87 --edge 50 --device 1 --models 'GCN' --embed --attack

# python tsne.py --dataset dblp --alpha 0.1 --beta 1 --gamma 8 --node 32 --edge 24 --epochs 500 --device 3 --models 'GCN' --embed --attack