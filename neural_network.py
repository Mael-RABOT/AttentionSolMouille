import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision import transforms

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d()
        self.maxpool1 = torch.nn.MaxPool2d()
        self.conv2 = torch.nn.Conv2d()
        self.maxpool2 = torch.nn.MaxPool2d()
