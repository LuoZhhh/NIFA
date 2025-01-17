# Implementations for NIFA

This repository includes the implementations for our paper at NeurIPS 2024: [***Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections.***](https://arxiv.org/abs/2406.03052)

<img src="https://github.com/CGCL-codes/NIFA/blob/main/framework.png" alt="Framework of NIFA">

## Environments

Python 3.8.12

Packages:

```
dgl==0.6.1
dgl_cu110==0.6.1
numpy==1.21.4
torch==1.7.1+cu110
tqdm==4.62.3
```

Run the following code to install all required packages.

```
> pip install -r requirements.txt
```

## Datasets & Processed files

- Due to size limitation, the processed datasets are stored in  [google drive](https://drive.google.com/file/d/1WJYj8K3_H3GmJg-RZeRsJ8Z64gt3qCnq/view?usp=drive_link) as `data.zip`. The datasets include Pokec-z, Pokec-n and DBLP. 

- Download and unzip the `data.zip`, and the full repository should be as follows:

  ```
  .
  ├── code
  │   ├── attack.py
  │   ├── main.py
  │   ├── model.py
  │   ├── run.sh
  │   └── utils.py
  ├── data
  │   ├── dblp.bin
  │   ├── pokec_n.bin
  │   └── pokec_z.bin
  ├── readme.md
  └── requirements.txt
  ```

## Run the codes

### Evaluation of classic GNN models

All arguments are properly set below for reproducing our results on four classic GNN models. 

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 2 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'
```

### Evaluation of fair GNN models

Since FairGNN, FairVGNN, and FairSIN have been independently open-sourced, we have not included a unified integration of these models in our repository yet. For the convenience of reproducing our results on these models, we summarize their official repositories below:

| MODEL      | Repository |
| ---------- | ---------- |
| FairGNN    | [Github](https://github.com/EnyanDai/FairGNN)     |
| FairVGNN   | [Github](https://github.com/yuwvandy/FairVGNN)    |
| FairSIN    | [Github](https://github.com/BUPT-GAMMA/FairSIN)   |

Since the dataset processing methods of FairGNN, FairVGNN, and FairSIN are highly similar, after generating the poisoned graph using NIFA, you can use the following code to process the dataset and make it compatible with their repositories for subsequent evaluations:

```
glist, _ = dgl.load_graphs(f'./output/{args.dataset}_poisoned.bin')  # load poisoned graph file
g = glist[0]

idx_train = torch.where(g.ndata['train_index'])[0]
idx_val = torch.where(g.ndata['val_index'])[0]
idx_test = torch.where(g.ndata['test_index'])[0]
index_split = {'train_index': idx_train,
                'val_index': idx_val,
                'test_index': idx_test}
features = g.ndata['feature']
labels = g.ndata['label']
sens = g.ndata['sensitive'] 
idx_train, idx_val, idx_test = index_split['train_index'], index_split['val_index'], index_split['test_index']
adj = sp.coo_matrix((np.ones(g.edges()[0].shape[0]), (g.edges()[0], g.edges()[1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
idx_sens_train = idx_train
```

## Licenses

This project is licensed under CC BY-NC-ND 4.0. To view a copy of this license, please visit http://creativecommons.org/licenses/by-nc-nd/4.0/

## BibTeX

If you like our work and use the model for your research, please cite our work as follows:

```bibtex
@inproceedings{luo2024nifa,
author = {Luo, Zihan and Huang, Hong and Zhou, Yongkang and Zhang, Jiping and Chen, Nuo and Jin, Hai},
title = {Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections},
booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
year = {2024},
month = {October}
}
``` 
