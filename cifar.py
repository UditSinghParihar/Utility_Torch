import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(), 
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
	img = img/2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# print([classes[labels[j]] for j in range(4)])
# imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# dataiter = iter(trainloader)
# inputs, labels = dataiter.next()

# outputs = net(inputs)

PATH = "./cifar_net.pth"
def train():
	for epoch in range(2):
		runningLoss = 0.0

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)

			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			runningLoss += loss.item()
			if(i%2000 == 1999):
				print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, runningLoss / 2000))
				runningLoss = 0.0

	print("Finished Training")
	torch.save(net.state_dict(), PATH)


train()

dataiter = iter(trainloader)
images, labels = dataiter.next()

print([classes[labels[j]] for j in range(4)])
imshow(torchvision.utils.make_grid(images))

netInfer = Net()
netInfer.load_state_dict(torch.load(PATH))
outputs = netInfer(images)

_, predicted = torch.max(outputs, 1)
print([classes[predicted[j]] for j in range(4)])
print(outputs.shape, predicted.shape)

def test():
	correct = 0
	total = 0

	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = netInfer(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print("Accuracy on 10000 images: {}".format(100*correct/total))

test()