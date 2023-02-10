import torchaudio.datasets

import neural_network
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchaudio.datasets import SPEECHCOMMANDS

from tqdm import tqdm
import neural_network

class Application:
    def __init__(self, lr=0.00001, epoch=25, batch_size=32, model_path="./save/model_save.asm"):
        self.model = None
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_path = model_path
        self.train_set = SPEECHCOMMANDS(root="./datasets", download=True, subset="training")
        self.test_set = SPEECHCOMMANDS(root="./datasets", download=True, subset="testing")
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("Model has been saved")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print("Model has been saved")

    def forward(self, input):
        return self.model.forward(input)
