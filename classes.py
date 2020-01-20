import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def fun1():
	class Out(object):
		"""docstring for Out"""
		def __init__(self):
			self.data = np.arange(20).reshape(4, 5)

		def __getitem__(self, index):
			return self.data[index]


	obj = Out()
	print(obj[2])

	class Rectangle(object):
		"""docstring for Rectangle"""
		def __init__(self, leng, width):
			self.width = width
			self.leng = leng
			
		def area(self):
			return self.width * self.leng

	class Square(Rectangle):
		"""docstring for Square"""
		def __init__(self, leng):
			# super().__init__(leng, leng)
			self.size = leng

	sq = Square(5)
	print(sq.area())


def fun2():
	inputSize = 5
	outSize = 2
	batchSize = 30
	dataSize = 100

	class RandomDataset(Dataset):
		"""docstring for RandomDataset"""
		def __init__(self, size, length):
			self.len = length
			self.data = torch.randn(length, size)

		def __getitem__(self, index):
			return self.data[index]

		def __len__(self):
			return self.len

	randLoader = DataLoader(dataset=RandomDataset(inputSize, dataSize), batch_size=batchSize, shuffle=True)

	class Model(nn.Module):
		"""docstring for Model"""
		def __init__(self, inputSize, outSize):
			super(Model, self).__init__()
			self.fc = nn.Linear(inputSize, outSize)

		def forward(self, inp):
			output = self.fc(inp)
			print("\tIn Model: Input size: {} and Output size: {}".format(inp.size(), output.size()))

			return output

	model = Model(inputSize, outSize)
	if torch.cuda.device_count() > 1:
		print("Let's use {} GPUs!".format(torch.cuda.device_count()))
		model = nn.DataParallel(model)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	for data in randLoader:
		inp = data.to(device)
		output = model(inp)
		print("Outside Model: input size {} and output size {}".format(inp.size(), output.size()))


class Rect(object):
	"""docstring for Rect"""
	def __init__(self, leng, width):
		super(Rect, self).__init__()
		self.leng = leng
		self.width = width

	def __call__(self, height):
		return self.leng*self.width*height

rect = Rect(2, 5)
print("Volume of cuboid is {}".format(rect(4)))