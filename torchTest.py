import torch
import torch as t
import os


def fun1():
	print(torch.cuda.is_available())
	print(torch.empty(5, 3))
	print(torch.rand(5, 3))

	cmd1, cmd2 = "pwd", "ls /home/"

	res1 = os.system(cmd1)
	# res2 = os.system(cmd2)
	print(res1)

	print(torch.tensor([4, 5.3]))

	t1 = torch.tensor([10, -13.3])
	print(torch.rand_like(t1, dtype=torch.float))

	t2 = t1.new_ones(2, 3, dtype=torch.double)
	print(t2, t2.size())
	print(t2.shape)

	t3 = torch.arange(15).view(3, 5).float()
	t4 = torch.ones(3, 5)
	print(torch.add(t3 ,t4))

	t5 = t3.view(15)
	print(t5, t5.shape)

	print(torch.is_tensor(t3))

	print(torch.arange(3).dtype)

	t6 = t3.cuda()
	print(t3.device, t6.device)

	print(t3.numpy())


def fun2():
	x = t.arange(15).view(3, 5).float()
	print(x, x.shape, x.device)

	x = x.to("cuda")
	device = t.device("cuda")
	y = t.ones_like(x, device=device)
	z = x + y
	print(z)
	z2 = z.to("cpu")
	print(z2)


import numpy as np


def fun3():
	x = torch.ones(2, 2, requires_grad=True)
	print(x, x.device)
	x2 = np.array([[2.0, -1], [1, 10]])
	x3 = t.from_numpy(x2)
	x3.requires_grad_(True)

	y = x3+2
	print(y, y.grad_fn)
	z = y*y*3
	out = z.mean()
	print(z, out, out.requires_grad)

	out.backward()
	print(x3.grad)


def fun4():
	x = np.array([2.0, 3, -1]).reshape(3, 1)
	x = t.from_numpy(x)
	x.requires_grad_(True)
	y = (x+2)**3
	z = y.mean()
	z.backward()
	print(x.grad)

	inputs = np.array([[0.52, 1.12,  0.77],
					   [0.88, -1.08, 0.15],
					   [0.52, 0.06, -1.30],
					   [0.74, -2.49, 1.39]])

	weights = np.array([0.0, 0.0, 0.0])

	print(inputs.shape, weights.shape)
	print(np.dot(inputs, weights))


def fun5():
	x = torch.tensor([[-1], [3.0]], requires_grad=True)
	A = torch.tensor([[1, 0], [1, 1.0]])
	y = torch.mm(A, x**2)

	v = torch.tensor([10.0, 30]).view(2, 1)
	y.backward(v)
	print(x.grad)

	input2 = torch.randn(1, 256, 256)
	input2 = input2.unsqueeze(0)
	print(input2.shape)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def fun6():
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()

			self.conv1 = nn.Conv2d(1, 6, 3)
			self.conv2 = nn.Conv2d(6, 16, 3)
			self.fc1 = nn.Linear(16 * 6 * 6, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

		def forward(self, x):
			x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
			x = F.max_pool2d(F.relu(self.conv2(x)), 2)
			x = x.view(-1, self.num_flat_features(x))
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x

		def num_flat_features(self, x):
			size = x.size()[1:]
			num_features = 1
			for s in size:
				num_features *= s
			return num_features


	net = Net()
	print(net)

	input1 = torch.randn(1, 1, 32, 32)
	out = net(input1)
	# print(out, out.requires_grad)

	params = list(net.parameters())
	print(len(params), params[0].shape, params[9].shape)
	print(net.parameters())

	optimizer = optim.SGD(net.parameters(), lr=0.01)
	criterion = nn.MSELoss()

	target = torch.randn(10).view(1, -1)
	print(target.shape, out.shape)

	for i in range(200):
		out = net(input1)
		optimizer.zero_grad()
		loss = criterion(out, target)
		print(loss)
		loss.backward()
		optimizer.step()


from torch.utils.data import Dataset, DataLoader


def fun7():
	inputSize = 5
	outSize = 2
	batchSize = 30
	dataSize = 100

	class RandomDataset(Dataset):
		"""docstring for RandomDataset"""
		def __init__(self, size, length):
			self.len = length
			self.data = torch.arange(length*size).view(length, size)

		def __getitem__(self, index):
			return self.data[index]

		def __len__(self):
			return self.len

	randLoader = DataLoader(dataset=RandomDataset(inputSize, dataSize), batch_size=batchSize, shuffle=False)

	dataiter = iter(randLoader)
	data2 = dataiter.next()
	print(data2)


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def fun8():
	landMarks = pd.read_csv('data/faces/face_landmarks.csv')
	n = 65
	imgName = landMarks.iloc[n, 0]
	landMarks = landMarks.iloc[n, 1:].as_matrix()
	landMarks = landMarks.astype('float').reshape(-1, 2)
	print("First 4 landMarks: {}".format(landMarks[:4]))


	def showLandmarks(image, landmarks):
		"""Show image with landmarks"""
		plt.imshow(image)
		plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
		plt.pause(0.001)
		plt.show()


	showLandmarks(io.imread(os.path.join('data/faces', imgName)), landMarks)


X1 = torch.arange(5.0, 10000, 5)
Y1 = torch.ones(1999)
X2 = torch.arange(7.0, 10000, 7)
Y2 = torch.zeros(1428)

X = torch.cat((X1, X2), 0)
Y = torch.cat((Y1, Y2), 0)

xMax, xMin = torch.max(X), torch.min(X)
X = (X - xMin)/(xMax - xMin)

w, b = torch.tensor([0.1]), torch.tensor([0.8])
w.requires_grad_(True)
b.requires_grad_(True)


for _ in range(1000):
	Z = w*X + b
	pred = torch.sigmoid(Z)

	num = Z.shape[0]
	cost = (-1/num) * torch.sum((Y)*(torch.log(pred)) + (1-Y)*(torch.log(1-pred)))
	print(cost)
	cost.backward()

	lr = torch.tensor(0.001)

	w.data.sub_(lr*w.grad.data)
	b.data.sub_(lr*b.grad.data)

with torch.no_grad():
	x = torch.tensor([15.0, 63, 21, 35, 255, 700])
	xMax, xMin = torch.max(x), torch.min(x)
	x = (x - xMin)/(xMax - xMin)
	inf = torch.sigmoid(w*x + b)
	print(inf)