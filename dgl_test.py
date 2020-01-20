import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from dgl.data import MiniGCDataset
import dgl.function as fn
import torch.optim as optim
from torch.utils.data import DataLoader


def fun1():
	def buildGraph():
		g = dgl.DGLGraph()
		g.add_nodes(34)

		edgeList = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
			(4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
			(7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
			(10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
			(13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
			(21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
			(27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
			(31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
			(32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
			(32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
			(33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
			(33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
			(33, 31), (33, 32)]
		src, dst = tuple(zip(*edgeList))
		
		g.add_edges(src, dst)
		g.add_edges(dst, src)

		return g


	def gcnMesg(edges):
		return {'msg' : edges.src['h']}


	def gcnReduce(nodes):
		return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}


	class GCNLayer(nn.Module):
		def __init__(self, inFeats, outFeats):
			super(GCNLayer, self).__init__()
			self.linear = nn.Linear(inFeats, outFeats)
			
		def forward(self, g, inputs):
			g.ndata['h'] = inputs
			g.send(g.edges(), gcnMesg)
			g.recv(g.nodes(), gcnReduce)
			h = g.ndata.pop('h')

			return self.linear(h)


	class GCN(nn.Module):
		def __init__(self, inFeats, hiddenSize, numClasses):
			super(GCN, self).__init__()
			self.gcn1 = GCNLayer(inFeats, hiddenSize)
			self.gcn2 = GCNLayer(hiddenSize, numClasses)

		def forward(self, g, inputs):
			h = self.gcn1(g, inputs)
			h = torch.relu(h)
			h = self.gcn2(g, h)

			return h


	G = buildGraph()
	print("Num edges: {} and Num nodes: {}".format(G.number_of_edges(), G.number_of_nodes()))

	G.ndata['feat'] = torch.eye(34)
	# print(G.nodes[2].data['feat'])
	print(type(G.ndata['feat']))

	net = GCN(34, 5, 2)

	inputs = torch.eye(34)
	labeledNodes = torch.tensor([0, 33])
	labels = torch.tensor([0, 1])

	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	allLogits = []

	for epoch in range(30):
		logits = net(G, inputs)
		allLogits.append(logits.detach())
		logP = F.log_softmax(logits, 1)

		loss = F.nll_loss(logP[labeledNodes], labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print("Epoch: {} | Loss: {:.4f}".format(epoch, loss.item()))


def draw(g):
	nx.draw(g.to_networkx(), with_labels=True)
	plt.show()


def fun2():
	gNx = nx.petersen_graph()
	gDgl = dgl.DGLGraph(gNx)

	plt.subplot(121)
	nx.draw(gNx, with_labels=True)
	plt.subplot(122)
	nx.draw(gDgl.to_networkx(), with_labels=True)
	plt.show()

	g2 = dgl.DGLGraph()
	g2.add_nodes(8)

	for i in range(1, 4):
		g2.add_edge(i, 0)

	src = list(range(5, 8)); dst = [0]*3
	g2.add_edges(src, dst)

	draw(g2)
	# nx.draw(g2.to_networkx(), with_labels=True)
	# plt.show()

	g2.clear()
	g2.add_nodes(10)
	src = torch.tensor(list(range(1, 10)))
	g2.add_edges(src, 0)

	draw(g2)


def fun3():
	x = torch.randn(10, 3)
	g = dgl.DGLGraph()
	g.add_nodes(10)
	src = torch.tensor(list(range(1, 10)))
	g.add_edges(src, 0)

	g.ndata['x'] = x
	g.edata['w'] = torch.randn(9, 2)
	# print(g.ndata['x'])
	# print(g.nodes[0, 1, 2].data['x'])
	# print(g.number_of_edges())
	# print(g.edata['w'])
	# print(g.edges[[0, 1, 2]].data['w'])

	draw(g)
	print(g)


dataset = MiniGCDataset(80, 10, 20)
# print(type(dataset), len(dataset))

graphs, labels = list(zip(*dataset))

# for graph, label in zip(graphs, labels):
# 	print(graph, label)
# 	draw(graph)


def collate(samples):
	graphs, labels = map(list, zip(*samples))
	batchedGraph = dgl.batch(graphs)
	return batchedGraph, torch.tensor(labels)


msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
	accum = torch.mean(nodes.mailbox['m'], 1)
	return {'h': accum}


class NodeApplyModule(nn.Module):
	def __init__(self, inFeats, outFeats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(inFeats, outFeats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)

		return {'h' : h}


class GCN(nn.Module):
	def __init__(self, inFeats, outFeats, activation):
		super(GCN, self).__init__()
		self.applyMod = NodeApplyModule(inFeats, outFeats, activation)
		
	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(msg, reduce)
		g.apply_nodes(func=self.applyMod)

		return g.ndata.pop('h')


class Classfier(nn.Module):
	def __init__(self, inDim, hiddenDim, nClasses):
		super(Classfier, self).__init__()

		self.layers = nn.ModuleList([
			GCN(inDim, hiddenDim, F.relu),
			GCN(hiddenDim, hiddenDim, F.relu)])
		self.classify = nn.Linear(hiddenDim, nClasses)

	def forward(self, g):
		h = g.in_degrees().view(-1, 1).float()
		for conv in self.layers:
			h = conv(g, h)
		g.ndata['h'] = h
		hg = dgl.mean_nodes(g, 'h')

		return self.classify(hg)


trainset = MiniGCDataset(320, 10, 20)
testset = MiniGCDataset(80, 10, 20)
dataloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)

model = Classfier(1, 256, trainset.num_classes)
lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

PATH = "checkpoints/graph_classify.pth"
def train():
	model.train()
	epochLosses = []
	for epoch in range(80):
		epochLoss = 0
		for iter, (bg, label) in enumerate(dataloader):
			prediction = model(bg)
			loss = lossFunc(prediction, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epochLoss += loss.detach().item()
		epochLoss /= (iter + 1)
		print('Epoch {}, loss {:.4f}'.format(epoch, epochLoss))
		epochLosses.append(epochLoss)

	torch.save(model.state_dict(), PATH)

	plt.title("Cross Entropy Loss averaged over minibatches")
	plt.plot(epochLosses)
	plt.show()

# train()

netInfer = Classfier(1, 256, trainset.num_classes)
netInfer.load_state_dict(torch.load(PATH))


model.eval()
with torch.no_grad():
	testX, testY = map(list, zip(*testset))
	testBg = dgl.batch(testX)
	testY = torch.tensor(testY).float().view(-1, 1)

	probsY = torch.softmax(netInfer(testBg), 1)
	sampledY = torch.multinomial(probsY, 1)
	argmaxY = torch.max(probsY, 1)[1].view(-1, 1)

	accSam = 100 * (testY == sampledY.float()).sum().item() / len(testY)
	accArg = 100 * (testY == argmaxY.float()).sum().item() / len(testY)
	print("Sampled Accuracy: {:.4f}% Argmax accuracy: {:.4f}".format(accSam, accArg))