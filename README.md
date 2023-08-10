# DropREGCN
This is the PyTorch implementation of our paper: "DropRE-GCN: Decoupled Graph Neural Network with DropRE Sparsification"

## Requirement
- pytorch >= 1.8.0

- torch-geometric

- torchmetrics

Note that the versions of PyTorch and PyTorch Geometric should be compatible.

## Datsetset
The `data` folder contains five benchmark datasets (Cora, Citeseer, Pubmed, CS, Photo). 

Cora, CiteSeer, and PubMed are pre-downloaded from [`torch_geometric.datasets`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid).

CS are pre-downloaded from [`torch_geometric.datasets`]( https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Coauthor)

Photo are pre-downloaded from [`torch_geometric.datasets`](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/amazon.html)

For the Planetoid dataset we use the same full-supervised setting as [FastGCN](https://github.com/matenure/FastGCN).

For CS and Photo dataset we use the same semi-supervised setting as [GNN-benchmark](https://github.com/shchur/gnn-benchmark)
## Example
For Cora dataset, please run:
```
python main.py --dataset=Cora --lr=0.002 --weight_decay=5e-2 --k=10 --dropout=0.4
```
For CiteSeer dataset, please run:
```
python main.py --dataset=CiteSeer --lr=0.013 --weight_decay=5e-2 --k=6 --dropout=0.6
```
For PubMed dataset, please run:
```
python main.py --dataset=PubMed --lr=0.019 --weight_decay=5e-4 --k=6 --dropout=0.6
```
For CS dataset, please run:
```
python main.py --dataset=CS --lr=0.005 --weight_decay=5e-2 --k=7 --dropout=0.7
```
For Photo dataset, please run:
```
python main.py --dataset=Photo --lr=0.002 --weight_decay=5e-2 --k=5 --dropout=0.4
```
