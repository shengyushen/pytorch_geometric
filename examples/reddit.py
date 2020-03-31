import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv

import time

def timeit(func):
  start = time.time()
  res = func()
  end= time.time()
  print("runtime {}".format(start-end))
  return res

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SAGE')
args = parser.parse_args()
assert args.model in ['SAGE', 'GAT']
print("model : {}".format(args.model))

# SSY downloading dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]
loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=10000,
                         shuffle=True, add_self_loops=True)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=False, concat=concat)
        self.conv2 = SAGEConv(16, out_channels, normalize=False, concat=concat)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(
            self.conv1((x, None), data.edge_index, size=data.size,
                       res_n_id=data.res_n_id))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size,
                       res_n_id=data.res_n_id)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = F.elu(
            self.conv1((x, x[block.res_n_id]), block.edge_index,
                       size=block.size))
        x = F.dropout(x, p=0.6, training=self.training)
        block = data_flow[1]
        x = self.conv2((x, x[block.res_n_id]), block.edge_index,
                       size=block.size)
        return F.log_softmax(x, dim=1)

# SSY 4 GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# SSY 4 CPU
device = torch.device('cpu')
print("cuda is {}".format(torch.cuda.is_available()))
Net = SAGENet if args.model == 'SAGE' else GATNet
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    xx = 0
    # SSY shared down load x
    # this cost ~100ms
    data_x_dev = data.x.to(device)

    l = []
    for data_flow in loader(data.train_mask):
      l.append(data_flow)
      
    start = time.time()
    for data_flow in l:
        #print(type(data_flow))
        print("batch_size {} ".format(data_flow.batch_size))
        end= time.time()
        print("runtime {}".format(end-start))
        print(xx)
        xx = xx+1
        # SSY init grad
        optimizer.zero_grad()
        # SSY data_flow gose to gpu seems to be bottle neck
        #out = model(data.x.to(device), data_flow.to(device))
        out = model(data_x_dev, data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        # SSY compute gradient
        loss.backward()
        # update parameter
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
        start = time.time()

    return total_loss / data.train_mask.sum().item()


def test(mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


#for epoch in range(1, 31):
for epoch in range(1, 4):
    #loss = timeit(train)
    loss = train()
    #test_acc = test(data.test_mask)
    #print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
    #    epoch, loss, test_acc))
