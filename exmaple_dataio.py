import torchvision
import torch
from torchvision.transforms import *
from utils import *

ensure
training_data = torchvision.datasets.FashionMNIST('..\\data',train=True,download=True,transform=ToTensor())
testing_data = torchvision.datasets.FashionMNIST('..\\data',train=False,download=True,transform=ToTensor())

print(training_data[0])