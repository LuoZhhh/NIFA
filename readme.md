# Implementation for IJCAI 2024 #3239

This repository includes the implementation for our paper: ***Are Your Models Still Fair? Adversarial Fairness Attacks on Graph Neural Networks via Node Injections.***

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

- Due to size limitation, the processed datasets are stored in  [google drive](https://drive.google.com/file/d/1WJYj8K3_H3GmJg-RZeRsJ8Z64gt3qCnq/view?usp=drive_link) as `data.zip` with anomynous user information. The datasets include Pokec-z, Pokec-n and DBLP. 

- Download and unzip the `data.zip`, and the full repository should be as follows:

  ```
  .
  ├── code
  │   ├── attack.py
  │   ├── main.py
  │   ├── model.py
  │   ├── models
  │   ├── __pycache__
  │   ├── run.sh
  │   └── utils.py
  ├── data
  │   ├── dblp.bin
  │   ├── pokec_n.bin
  │   └── pokec_z.bin
  ├── NIFA.zip
  └── requirements.txt
  ```

## Run the codes

All arguments are properly set in advance in the script files `run.sh` for reproducing our results. 

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 2 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'
```
