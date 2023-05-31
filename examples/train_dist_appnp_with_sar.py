# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union, Dict
from argparse import ArgumentParser
import os
import logging
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
import pdb

from dgl.nn.pytorch.conv import APPNPConv

import sar


parser = ArgumentParser(
    description="GNN training on node classification tasks in homogeneous graphs")


parser.add_argument(
    "--partitioning-json-file",
    type=str,
    default="",
    help="Path to the .json file containing partitioning information "
)
parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it ')


parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi'],
                    help='Communication backend to use '
                    )

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)


parser.add_argument('--train-iters', default=100, type=int,
                    help='number of training iterations ')

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate"
)


parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers ')

parser.add_argument('--hidden-layer-dim', default=[64], type=int, nargs="+",
                    help='Dimension of GNN hidden layer')
parser.add_argument(
    "--k", type=int, default=10, help="Number of propagation steps"
)
parser.add_argument(
    "--alpha", type=float, default=0.1, help="Teleport Probability"
)
parser.add_argument(
    "--in-drop", type=float, default=0.5, help="input feature dropout"
)
parser.add_argument(
    "--edge-drop", type=float, default=0.5, help="edge propagation dropout"
)


class APPNP(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        hiddens,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        print(in_feats, hiddens)
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, features):
        # pdb.set_trace()
        # prediction step
        h = features
        # h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        # h = self.layers[-1](self.feat_drop(h))
        h = self.layers[-1](h)
        # propagation step
        h = self.propagate(self.g, h)
        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main():
    args = parser.parse_args()
    print('args', args)

    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank,
                         args.world_size,
                         master_ip_address,
                         args.backend)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, device)

    # Obtain train,validation, and test masks
    # These are stored as node features. Partitioning may prepend
    # the node type to the mask names. So we use the convenience function
    # suffix_key_lookup to look up the mask name while ignoring the
    # arbitrary node type
    masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data.node_features,
                                             mask_name)
        masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)

    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   'label').long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    print(labels, partition_data.node_features)
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    num_labels = num_labels.item()

    features = sar.suffix_key_lookup(partition_data.node_features, 'feat').to(device)
    full_graph_manager = sar.construct_full_graph(partition_data).to(device)

    #We do not need the partition data anymore
    # del partition_data

    dgl.random.seed(123)
    torch.random.manual_seed(123)
    gnn_model = APPNP(
                full_graph_manager,
                features.size(1),
                args.hidden_layer_dim,
                num_labels,
                F.relu,
                args.in_drop,
                args.edge_drop,
                args.alpha,
                args.k)
    dgl.random.seed(123)
    torch.random.manual_seed(123)
    gnn_model.reset_parameters()
    print('model', gnn_model)
    # Synchronize the model parmeters across all workers
    # for k,v in gnn_model.state_dict().items():
    #     v.data.zero_()

    sar.sync_params(gnn_model)

    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=5e-4)
    for train_iter_idx in range(args.train_iters):
        # Train
        t_1 = time.time()
        logits = gnn_model(features)
        loss = F.cross_entropy(logits[masks['train_indices']],
                               labels[masks['train_indices']], reduction='sum')/n_train_points

        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(gnn_model)
        optimizer.step()
        train_time = time.time() - t_1

        # Calculate accuracy for train/validation/test
        results = []
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            n_correct = (logits[masks[indices_name]].argmax(1) ==
                         labels[masks[indices_name]]).float().sum()
            results.extend([n_correct, masks[indices_name].numel()])

        acc_vec = torch.FloatTensor(results)
        # Sum the n_correct, and number of mask elements across all workers
        sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)
        (train_acc, val_acc, test_acc) =  \
            (acc_vec[0] / acc_vec[1],
             acc_vec[2] / acc_vec[3],
             acc_vec[4] / acc_vec[5])
        val_acc = evaluate(gnn_model, features, labels, masks['val_indices'])

        result_message = (
            f"iteration [{train_iter_idx}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"train loss={loss:.4f}, "
            f"Accuracy: "
            f"train={train_acc:.4f} "
            f"valid={val_acc:.4f} "
            f"test={test_acc:.4f} "
            f" | train time = {train_time} "
            f" |"
        ])
        print(result_message, flush=True)


if __name__ == '__main__':
    main()