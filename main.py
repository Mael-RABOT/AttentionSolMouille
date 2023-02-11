import torch
import torchvision
import gradio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from tkinter import *

class Game:
    def __int__(self):
        self.window = Tk()
        self.size = (1920, 1080)

        self.window.title("Attention sol mouillé")
        self.window.geometry(f"{self.size[0]}x{self.size[0]}")

    def start_window(self):
        self.window.mainloop()

test = Game()
