import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))])


trainset = torchvision.datasets.FashionMNIST('./dataFMNIST',
	download=True,
	train=True,
	transform=transform)
testset = torchvision.datasets.FashionMNIST('./dataFMNIST',
	download=True,
	train=False,
	transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										shuffle=False, num_workers=2)


classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
		'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def matplotlib_imshow(img, one_channel=False):
	if one_channel:
		img = img.mean(dim=0)
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

	# plt.show()


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_4')

dataiter = iter(trainloader)
images, labels = dataiter.next()

# matplotlib_imshow(images[0], one_channel=True)

imgGrid = torchvision.utils.make_grid(images)
matplotlib_imshow(imgGrid, one_channel=True)
writer.add_image('four_fashion_mnist_imgs', imgGrid)

writer.add_graph(net, images)

def select_n_random(data, labels, n=100):
	assert (len(data) == len(labels))

	perm = torch.randperm(len(data))
	return data[perm][:n], labels[perm][:n]

images, labels = select_n_random(trainset.data, trainset.targets)

class_labels = [classes[lab] for lab in labels]

features = images.view(-1, 28*28)
# writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1), global_step=3)
# writer.close()


def images_to_probs(net, images):
	output = net(images)
	# convert output probabilities to predicted class
	_, preds_tensor = torch.max(output, 1)
	preds = np.squeeze(preds_tensor.numpy())
	return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
	preds, probs = images_to_probs(net, images)
	# plot the images in the batch, along with predicted and true labels
	fig = plt.figure(figsize=(12, 48))
	for idx in np.arange(4):
		ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
		matplotlib_imshow(images[idx], one_channel=True)
		ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
			classes[preds[idx]],
			probs[idx] * 100.0,
			classes[labels[idx]]),
					color=("green" if preds[idx]==labels[idx].item() else "red"))
	# plt.show()
	return fig


running_loss = 0.0

for epoch in range(2):
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data

		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if (i%1000 == 999):
			writer.add_scalar('training_loss', running_loss/1000, epoch*len(trainloader) + i)

			writer.add_figure('predictions vs actuals', plot_classes_preds(net, inputs, labels),
								global_step=epoch*len(trainloader)+i)
			print("Training loss {}. Global step {}".format(running_loss/1000, epoch*len(trainloader)+i))
		running_loss = 0.0
print("Finished Training")


class_probs = []
class_preds = []
gt = []
with torch.no_grad():
	for data in testloader:
		images, labels = data
		output = net(images)
		class_probs_batch = [F.softmax(el, dim=0) for el in output]
		_, class_preds_batch = torch.max(output, 1)

		class_probs.append(class_probs_batch)
		class_preds.append(class_preds_batch)
		gt.append(labels)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)
gt_preds = torch.cat(gt)

def addPRCurve(class_index, test_probs, test_preds, global_step=0):
	tensorboard_preds = gt_preds == class_index
	tensorboard_probs = test_probs[:, class_index]

	writer.add_pr_curve(classes[class_index],
						tensorboard_preds,
						tensorboard_probs,
						global_step=global_step)
	writer.close()

# plot all the pr curves
for i in range(len(classes)):
	addPRCurve(i, test_probs, test_preds)