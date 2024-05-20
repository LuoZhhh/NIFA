
# python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

# python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

# python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'


python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 1 --mode 'degree' --models 'GCN'

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --mode 'degree' --models 'GCN'

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 1 --mode 'degree' --models 'GCN'