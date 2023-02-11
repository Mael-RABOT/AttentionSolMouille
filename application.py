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

class Application:
    def __init__(self, lr=0.00001, epochs=400, batch_size=32, model_path="./save/model_save.astm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = neural_network.NeuralNetwork().to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.train_set = torchaudio.datasets.SPEECHCOMMANDS(root="./datasets", download=True, subset="training")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))
        self.test_set = torchaudio.datasets.SPEECHCOMMANDS(root="./datasets", download=True, subset="testing")
        self.train_loader = None

    def get_label(self, index):
        return self.labels[index]

    def get_index(self, label):
        return torch.tensor(self.labels.index(label))

    def pad_sequence(self, batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.get_index(label)]
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)
        return tensors, targets

    def load_trainloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def train_model(self):
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
        print("sukssèsfoul trèning")
        return 0

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("modèle az bin saivèd")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print("modèle az bin laudid")

    def forward(self, input):
        return self.model.forward(input)

app = Application(epochs=25)
app.load_trainloader()
app.train_model()
app.save_model()