import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torchsummary import summary
from dgl.data import citation_graph as citegrh
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt


gcnMsg = fn.copy_src(src='h', out='m')
gcnReduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
	def __init__(self, inFeats, outFeats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(inFeats, outFeats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		if self.activation is not None:
			h = self.activation(h)

		return {'h' : h}


class GCN(nn.Module):
	def __init__(self, inFeats, outFeats, activation):
		super(GCN, self).__init__()
		self.applyMod = NodeApplyModule(inFeats, outFeats, activation)
		
	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcnMsg, gcnReduce)
		g.apply_nodes(func=self.applyMod)

		return g.ndata.pop('h')


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.gcn1 = GCN(1433, 16, F.relu)
		self.gcn2 = GCN(16, 7, None)

	def forward(self, g, features):
		x = self.gcn1(g, features)
		print("Features size: ", x.shape)
		x = self.gcn2(g, x)
		print("Features size: ", x.shape)
		
		return x


net = Net()
print(net)


def loadCoraData():
	data = citegrh.load_cora()
	
	features = th.FloatTensor(data.features)
	labels = th.LongTensor(data.labels)
	trainMask = th.BoolTensor(data.train_mask)
	testMask = th.BoolTensor(data.test_mask)

	g = data.graph
	g.remove_edges_from(nx.selfloop_edges(g))
	g = DGLGraph(g)
	g.add_edges(g.nodes(), g.nodes())
	
	return g, features, labels, trainMask, testMask


def evaluate(model, g, features, labels, mask):
	model.eval()
	with th.no_grad():
		logits = model(g, features)
		logits = logits[mask]
		labels = labels[mask]
		_, indices = th.max(logits, dim=1)
		correct = th.sum(indices == labels)

		return correct.item() * 1.0 / len(labels)


def draw(g):
	nx.draw(g.to_networkx(), with_labels=True)
	plt.show()


g, features, labels, trainMask, testMask = loadCoraData()
PATH = "checkpoints/gcn.pth"


def train():
	optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
	dur = []
	epochLosses = []
	accs = []

	for epoch in range(500):
		if epoch >= 3:
			t0 = time.time()

		net.train()
		logits = net(g, features)
		logp = F.log_softmax(logits, 1)
		loss = F.nll_loss(logp[trainMask], labels[trainMask])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch >= 3:
			dur.append(time.time() - t0)

		acc = evaluate(net, g, features, labels, testMask)
		epochLosses.append(loss.detach().item())
		accs.append(acc)

		print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(),
			acc, np.mean(dur)))


	th.save(net.state_dict(), PATH)
	
	plt.title("NLL loss over epochs")
	plt.plot(epochLosses)
	plt.show()


# train()

netInfer = Net()
netInfer.load_state_dict(th.load(PATH))

netInfer.eval()
with th.no_grad():
	acc = evaluate(netInfer, g, features, labels, testMask)
	print("End Accuracy: {}".format(acc))


print(type(g), type(features), features.shape)
print(g.number_of_edges(), g.number_of_nodes())
print(th.sum(trainMask), th.sum(testMask))
# draw(g)