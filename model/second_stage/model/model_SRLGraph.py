import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        # nn.init.kaiming_normal_(self.linear.weight, a=0, mode='fan_out', nonlinearity='relu')

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(300, 300)
        self.layer2 = GCNLayer(300, 300)
        self.layer3 = GCNLayer(300, 300)


    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = F.sigmoid(self.layer3(g, x))
        return x

class GCN_4_3_relu(nn.Module):
    def __init__(self):
        super(GCN_4_3_relu, self).__init__()
        self.layer1 = GCNLayer(300, 300)
        self.layer2 = GCNLayer(300, 300)
        self.layer3 = GCNLayer(300, 300)
        self.layer4 = GCNLayer(300, 3)

    def forward(self, g, features):
        x = F.softmax(self.layer1(g, features))
        x = F.softmax(self.layer2(g, x))
        x = F.relu(self.layer3(g, x))
        x = F.log_softmax(self.layer4(g, x))
        return x

class GCN_4_3_leaky_relu(nn.Module):
    def __init__(self):
        super(GCN_4_3_leaky_relu, self).__init__()
        self.layer1 = GCNLayer(300, 300)
        self.layer2 = GCNLayer(300, 300)
        self.layer3 = GCNLayer(300, 300)
        self.layer4 = GCNLayer(300, 3)

    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.log_softmax(self.layer4(g, x))
        return x

class GCN_4(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(300, 300)
        self.layer2 = GCNLayer(300, 300)
        self.layer3 = GCNLayer(300, 300)
        self.layer4 = GCNLayer(300, 300)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = F.relu(self.layer3(g, x))
        x = F.sigmoid(self.layer4(g, x))
        return x

class GCN_4_3(nn.Module):
    def __init__(self):
        super(GCN_4_3, self).__init__()
        self.layer1 = GCNLayer(300, 300)
        self.layer2 = GCNLayer(300, 300)
        self.layer3 = GCNLayer(300, 300)
        self.layer4 = GCNLayer(300, 3)

    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.log_softmax(self.layer4(g, x))
        return x

class GCN_6_3_leaky_relu(nn.Module):
    def __init__(self):
        super(GCN_6_3_leaky_relu, self).__init__()
        self.layer1 = GCNLayer(300, 200)
        self.layer2 = GCNLayer(200, 150)
        self.layer3 = GCNLayer(150, 100)
        self.layer4 = GCNLayer(100, 50)
        self.layer5 = GCNLayer(50, 25)
        self.layer6 = GCNLayer(25, 3)

    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.softmax(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.softmax(self.layer4(g, x))
        x = F.leaky_relu(self.layer5(g, x))
        x = F.log_softmax(self.layer6(g, x))
        return x

class SRLGrouph():
    def __init__(self):
        self.gcn = GCN()

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x