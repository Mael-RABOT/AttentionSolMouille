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
import torchaudio
from tqdm import tqdm
import neural_network
import signal
import readchar


class Application:
    def __init__(self, lr=0.00001, epochs=200, batch_size=64, model_path="./save/model_save_2.astm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = neural_network.NeuralNetwork().to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.train_set = torchaudio.datasets.SPEECHCOMMANDS(root="./datasets", download=True, subset="training")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))
        self.test_set = torchaudio.datasets.SPEECHCOMMANDS(root="./datasets", download=True, subset="testing")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.test_set)))
        self.train_loader = None
        self.test_loader = None

    def pad_sequence(self, batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.labels.index(label))]
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)
        return tensors, targets

    def load_trainloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def gros_train_sa_mere(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        i = 0
        while True:
            for audio, label in tqdm(self.train_loader):
                audio = audio.to(self.device)
                label = label.to(self.device)
                pred = self.model.forward(audio)
                loss = F.nll_loss(pred.squeeze(), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (i % self.epochs == 0 and i != 0):
                self.save_model()
                self.model = neural_network.NeuralNetwork().to(self.device)
                self.model_path += "I"
        print("sukss??sfoule tr??ning")
        return 0

    def train_model(self):
        losses = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for i in tqdm(range(self.epochs)):
            for audio, label in tqdm(self.train_loader):
                audio = audio.to(self.device)
                label = label.to(self.device)
                pred = self.model.forward(audio)
                loss = F.nll_loss(pred.squeeze(), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (i % 10 == 0 and i != 0):
                self.save_model()
        print("sukss??sfoule tr??ning")
        return 0

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("mod??le az bin saiv??d")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print("mod??le az bin laudid")

    def test(self):
        correct = 0
        total = 0
        for data, target in self.test_loader:
            pred = self.model.forward(data)
            pred = pred.argmax(dim=-1)
            correct += pred.squeeze().eq(target).sum().item()
            total += 1
        print(
            f"Accuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n")

    def execute_predict(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.unsqueeze(1)
        label = self.model.forward(waveform)
        return label[0][0].argmax(dim=0).item()

"""
def test(self):
    correct = 0
    total = 0
    for batch in self.train_set:
        data, merde, target, merde_2, merde_3 = batch
        print(data.shape)
        data = data.unsqueeze(1)
        data = data.to(self.device)
        data = transform(data)
        pred = self.model.forward(data)
        pred = pred[0][0]
        pred = pred.argmax(dim=-1)
        if self.labels[pred] == target:
            correct += 1
        total += 1
    print(f"{correct} correct sur {total} total")
"""