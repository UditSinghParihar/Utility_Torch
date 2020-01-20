import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, Entities
from sys import exit
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os.path as osp


def fun1():
	edgeIn = torch.Tensor([[1, 0, 1, 2, 2, 0], [0, 1, 2, 1, 3, 0]]).long()
	x = torch.Tensor([[-1.0], [0], [1], [-8.3]]).float()

	data = Data(x=x, edge_index=edgeIn)
	print(data)
	print(data.keys)
	print(data['x'])

	for key, item in data:
		print("{} -> {}".format(key, item))

	print("Directed: {} and Self loops: {}".format(data.is_directed(), data.contains_self_loops()))
	print("Nodes: {}, Features: {}, Edges: {}".format(data.num_nodes, data.num_node_features, data.num_edges))


def fun2():
	data2 = TUDataset("/tmp/ENZYMES", name="ENZYMES")
	print(len(data2), type(data2))
	print(data2.num_classes, data2.num_node_features)
	data2 = data2[50]
	print(data2.num_nodes, data2.num_edges, data2.num_node_features, data2.is_directed(), data2.contains_self_loops())
	print(data2)


def fun3():
	data3 = Planetoid("/tmp/Cora", name="Cora")
	device = torch.device("cuda")
	data3[0].to(device)

	print(data3, len(data3))
	data = data3[0]
	print(data.num_nodes, data.num_edges, data.num_node_features, data.is_directed(), data.contains_self_loops())
	print(data)
	print(data.x[0, 0:50])
	print(data.train_mask.sum())


def fun4():
	data4 = TUDataset("/tmp/ENZYMES", name="ENZYMES", use_node_attr=True)
	loader = DataLoader(data4, batch_size=32, shuffle=True)

	for data in loader:
		print(data)
		# print(data.batch[0:200])
		print(data.num_graphs)
		exit(1)


def fun5():
	data5 = Planetoid("/tmp/Cora", name="Cora")


	class Net(torch.nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.conv1 = GCNConv(data5.num_node_features, 16)
			self.conv2 = GCNConv(16, data5.num_classes)

		def forward(self, data):
			x, edgeIn = data.x, data.edge_index
			x = self.conv1(x, edgeIn)
			x = F.relu(x)
			x = F.dropout(x, training=self.training)
			x = self.conv2(x, edgeIn)

			return F.log_softmax(x, dim=1)



	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net().to(device)
	data = data5[0].to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


	def train():
		model.train()
		for epoch in range(200):
			optimizer.zero_grad()
			out = model(data)
			loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
			loss.backward()
			optimizer.step()


	def test():
		with torch.no_grad():
			model.eval()
			_, pred = model(data).max(dim=1)
			correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
			acc = correct/data.test_mask.sum().item()
			print("Accuracy: {:.4f}".format(acc))


	train()
	test()


name = 'MUTAG'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
dataset = Entities(path, name)
print(len(dataset), dataset.num_classes, dataset.num_node_features)
data = dataset[0]
print("Nodes: {}, Features: {}, Edges: {}".format(data.num_nodes, data.num_node_features, data.num_edges))
print("Directed: {}, Self loops: {}".format(data.is_directed(), data.contains_self_loops()))
