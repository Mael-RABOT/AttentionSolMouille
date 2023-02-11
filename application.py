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
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def train_model(self):
        losses = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for i in tqdm(range(self.epochs)):
            for audio, label in tqdm(self.train_loader):
                audio = audio.to(self.device)
                label = label.to(self.device)
                audio = transform(audio)
                pred = self.model.forward(audio)
                loss = F.nll_loss(pred.squeeze(), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("sukssèsfoule trèning")
        return 0

    def execute_predict(self, input):
        i = 0
        for audio, label in self.test_loader:
            if i > 0:
                return
            audio = audio.to(self.device)
            audio = transform(audio)
            print(len(label))
            print(self.model.forward(audio).shape)
            i += 1
        #pred = self.model.forward(audio)
        #print(pred)
        return 0

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("modèle az bin saivèd")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print("modèle az bin laudid")

    def number_of_correct(self, pred, target):
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(self, tensor):
        return tensor.argmax(dim=-1)

    def test(self):
        correct = 0
        for data, target in self.test_loader:
            print(data.shape)
            data = data.to(self.device)
            target = target.to(self.device)
            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = self.model.forward(data)

            pred = self.get_likely_index(output)
            correct += self.number_of_correct(pred, target)
        print(
            f"\nTest Epoch: {self.epochs}\tAccuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n")

    def execute_predict(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.unsqueeze(1)
        label = self.model.forward(waveform)
        print(self.labels)
        print(self.labels[label[0][0].argmax(dim=0)])

def handler(signum, frame):
    message = "Do you really want to exit y/n "
    print(message, end="", flush=True)
    res = readchar.readchar()
    if res == 'y':
        app.save_model()
        print("")
        exit(1)
    else:
        print("", end="\r", flush=True)
        print(" " * len(message), end="", flush=True)
        print("    ", end="\r", flush=True)





signal.signal(signal.SIGINT, handler)
app = Application(epochs=25)

waveform, sample_rate, label, speaker_id, utterance_number = app.train_set[0]
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)

app.load_trainloader()
#app.train_model()
#app.save_model()
app.load_model()
app.test()
app.execute_predict("./asm.wav")
#app.test_model()
#app.train_model()