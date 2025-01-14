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

from argparse import ArgumentParser
import dgl  # type:ignore
import torch
from ogb.nodeproppred import DglNodePropPredDataset  # type:ignore


parser = ArgumentParser(description="Graph partitioning for ogbn-arxiv and ogbn-products")

parser.add_argument(
    "--dataset-root",
    type=str,
    default="./datasets/",
    help="The OGB datasets folder "
)

parser.add_argument(
    "--dataset-name",
    type=str,
    default="ogbn-arxiv",
    choices=['ogbn-arxiv', 'ogbn-products'],
    help="Dataset name. ogbn-arxiv or ogbn-products "
)

parser.add_argument(
    "--partition-out-path",
    type=str,
    default="./partition_data/",
    help="Path to the output directory for the partition data "
)


parser.add_argument(
    '--num-partitions',
    default=2,
    type=int,
    help='Number of graph partitions to generate')


def main():
    args = parser.parse_args()
    dataset = DglNodePropPredDataset(name=args.dataset_name,
                                     root=args.dataset_root)
    graph = dataset[0][0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    labels = dataset[0][1].view(-1)
    split_idx = dataset.get_idx_split()

    def _idx_to_mask(idx_tensor):
        mask = torch.BoolTensor(graph.number_of_nodes()).fill_(False)
        mask[idx_tensor] = True
        return mask

    train_mask, val_mask, test_mask = map(
        _idx_to_mask, [split_idx['train'], split_idx['valid'], split_idx['test']])
    features = graph.ndata['feat']
    graph.ndata.clear()
    for name, val in zip(['train_mask', 'val_mask', 'test_mask', 'labels', 'features'],
                         [train_mask, val_mask, test_mask, labels, features]):
        graph.ndata[name] = val

    dgl.distributed.partition_graph(
        graph, args.dataset_name,
        args.num_partitions,
        args.partition_out_path,
        num_hops=1,
        balance_ntypes=train_mask,
        balance_edges=True)


if __name__ == '__main__':
    main()
