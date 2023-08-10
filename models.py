import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from utils import normalize_adj, sparse_diag, eliminate_negative, random_coauthor_amazon_splits


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class Data():
    def __init__(self, path, dataset, split, k, prob):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
            split (str): type of dataset split
            k (int) k-hop aggregation
            prob (float) The probability to trim adj
        """
        if dataset == "Photo":
            dataset = Amazon(root=path, name=dataset)
            data = random_coauthor_amazon_splits(dataset[0], dataset.num_classes, None)
        elif dataset == "Cora" or dataset == "CiteSeer" or dataset == "PubMed":
            dataset = Planetoid(root=path, name=dataset,split=split)
            data = dataset[0]
        elif dataset == "CS":
            dataset = Coauthor(path, dataset)
            data = random_coauthor_amazon_splits(dataset[0], dataset.num_classes, None)
        self.feature = data.x
        self.edge = data.edge_index
        self.label = data.y
        self.idx_train = torch.where(data.train_mask)[0]
        self.idx_val = torch.where(data.test_mask)[0]
        self.idx_test = torch.where(data.test_mask)[0]
        self.n_node = data.num_nodes
        self.n_edge = data.num_edges
        self.n_class = dataset.num_classes
        self.n_feature = dataset.num_features

        trim = torch.rand(k) > torch.pow((1 - prob), torch.arange(1, k + 1).float())

        self.adj = torch.sparse_coo_tensor(self.edge, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.adj = self.adj + sparse_diag(torch.ones(self.n_node))
        self.norm_adj = normalize_adj(self.adj, symmetric=True)
        self.norm_adjs = [self.norm_adj]
        for i in range(k - 1):
            adj = torch.sparse.mm(self.norm_adj, self.norm_adjs[i])
            if trim[i]:
                adj = NodeTrim(adj, self.norm_adjs[i])
            self.norm_adjs.append(adj)
        self.feature_diffused = [self.feature]
        for i in range(k):
            feature = torch.sparse.mm(self.norm_adjs[i], self.feature)
            self.feature_diffused.append(feature)


def NodeTrim(current, previous):
    mask = torch.sparse_coo_tensor(previous._indices(), torch.ones(previous._nnz()), previous.size())
    return eliminate_negative(current - torch.mul(current, mask))

class DropREGCN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, k, dropout, feature, norm_adjs):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            k (int): k-hop aggregation
            dropout (float): dropout rate
        """
        super(DropREGCN, self).__init__()

        self.k = k
        self.dropout = dropout
        self.feature = feature
        self.Lin1 = Linear(n_feature, 1)
        self.Lin2 = Linear(n_feature, n_hidden)
        self.Lin3 = Linear(n_hidden, n_class)
        self.norm_adjs = norm_adjs

    def forward(self, feature):
        """
        Args:
            feature (torch Tensor): feature input

        Returns:
            (torch Tensor): log probability for each class in label
        """
        pps = torch.stack(self.feature, dim=1)
        retain_score = self.Lin1(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pps = pps.to(device)
        out = torch.matmul(retain_score, pps).squeeze()
        out = self.Lin2(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = F.relu(out)
        out = self.Lin3(out)
        out = F.dropout(out, self.dropout, training=self.training)

        return F.log_softmax(out, dim=1)