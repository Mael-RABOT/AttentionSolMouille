import tkinter
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageTk
from tkinter import *
from random import randint
import subprocess
import sys
import signal
import readchar
import application
import microphone

ARR_SIZE = 10
EMPTY =  0
PLAYER = 1
ENEMY =  7
WALL =   5
WATER = 3
MOVE_SET = [0, 1, 2, 3]
MOVE_SET_DICT = {"30" : MOVE_SET[0], "5" : MOVE_SET[1], "15" : MOVE_SET[2], "22" : MOVE_SET[3]}
ENTITY_SET = {"player": PLAYER, "enemy": ENEMY}
PAUSED = True
CHARACT = None
OBSTACLES_NUM = 5


def change_pause_status() -> None:
    global PAUSED
    PAUSED = not PAUSED


def spawn_player(Map: list[list[int]], PlayerPos: list[int, int]=None) -> list[int, int]:
    if PlayerPos is not None:
        Map[PlayerPos[0]][PlayerPos[1]] = EMPTY
    PlayerPos = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    Map[PlayerPos[0]][PlayerPos[1]] = PLAYER
    return PlayerPos


def InputSelector(Window, app) -> int:
    Input = None
    while Input not in MOVE_SET:
        Window.update()
        microphone.start_record(1)
        Input = app.execute_predict("./assets/file1.wav")
        print(Input)
        if Input == 30 or Input == 5 or Input == 15 or Input == 22:
                Input = MOVE_SET_DICT[str(Input)]
    return Input


def MovePlayer(Map: list[list[int]], PlayerPos: list[int, int], PlayerAction: int, MainCanvas: tkinter.Canvas, Player: any) -> bool:
    PosBackup = [PlayerPos[i] for i in range(len(PlayerPos))]
    Map[PlayerPos[0]][PlayerPos[1]] = EMPTY
    if PlayerAction == 0 and PlayerPos[0] > 0:
        MainCanvas.move(Player, 0, -100)
        PlayerPos[0] = PlayerPos[0] - 1
    if PlayerAction == 1 and PlayerPos[0] < ARR_SIZE - 1:
        MainCanvas.move(Player, 0, +100)
        PlayerPos[0] = PlayerPos[0] +  1
    if PlayerAction == 2 and PlayerPos[1] > 0:
        MainCanvas.move(Player, -100, 0)
        PlayerPos[1] = PlayerPos[1] - 1
    if PlayerAction == 3 and PlayerPos[1] < ARR_SIZE - 1:
        MainCanvas.move(Player, 100, 0)
        PlayerPos[1] = PlayerPos[1] + 1
    else:
        return False
    Map[PlayerPos[0]][PlayerPos[1]] = PLAYER
    return True


def CheckPlot(Map: list[list[int]], Obstacle: list[list[int, int]], PlayerPos: list[int, int]) -> bool:
    if [PlayerPos[1], PlayerPos[0]] in Obstacle:
        return True
    return False


def AddObstacle(Obstacle: list[list[int, int]], PlayerPos: list[int, int]) -> None:
    ObstaclePos = [randint(1, ARR_SIZE - 2), randint(1, ARR_SIZE - 2)]
    while ObstaclePos in Obstacle and ObstaclePos == PlayerPos:
        ObstaclePos = [randint(1, ARR_SIZE - 2), randint(1, ARR_SIZE - 2)]
    Obstacle.append(ObstaclePos)


def game_loop(Window: tkinter.Tk, MainFrame: tkinter.Frame, enemy: bool=False, ObstacleNumber: int=OBSTACLES_NUM, app = None) -> None:
    Map = [[0 for j in range(ARR_SIZE)] for i in range(ARR_SIZE)]

    global PAUSED, CHARACT
    Game = True

    MainCanvas = Canvas(MainFrame, bg='blue', width=1000, height=1000, bd=0)
    MainCanvas.pack(fill=BOTH, expand=True, side='top')
    BackgroundImage = PhotoImage(file='assets/DungeonSafe.png')
    MainCanvas.create_image(500, 500, image=BackgroundImage, anchor=CENTER)
    if CHARACT == 'alex':
        PlayerImage = PhotoImage(file='assets/alex.png')
    elif CHARACT == 'clement':
        PlayerImage = PhotoImage(file='assets/clement.png')
    Player = MainCanvas.create_image(50, 50, image=PlayerImage)

    WaterImage = PhotoImage(file='assets/Plot.png')

    PlayerPos = spawn_player(Map)

    Obstacle = []

    WaterArr = [0 for i in range(ObstacleNumber + 1)]
    for i in range(ObstacleNumber):
        AddObstacle(Obstacle, PlayerPos)
        WaterArr[i] = MainCanvas.create_image(50, 50, image=WaterImage)
        MainCanvas.move(WaterArr[i], Obstacle[i][0] * 100, Obstacle[i][1] * 100)

    Map[Obstacle[1][0]][Obstacle[1][1]] = WATER

    while PlayerPos in Obstacle:
        PlayerPos = spawn_player(Map,PlayerPos)

    MainCanvas.move(Player, PlayerPos[1] * 100, PlayerPos[0] * 100)

    while Game:
        SkipTurn = False
        try:
            Window.state()
        except tkinter.TclError:
            exit()
        while PAUSED:
            Window.update()
            try:
                Window.state()
            except tkinter.TclError:
                exit()
        if not SkipTurn:
            PlayerAction = InputSelector(app)
            MovePlayer(Map, PlayerPos, PlayerAction, MainCanvas, Player)
            if CheckPlot(Map, Obstacle, PlayerPos):
                exit()


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


def launch_game() -> None:
    app = application.Application(model_path="./save/model_save_3.astm", epochs=5)
    app.load_trainloader()
    app.train_model()
    app.save_model()
    app.test()
    Window = Tk()
    Window.title("Attention sol mouillé")
    Window.geometry("1000x1100")
    Window.resizable(False, False)
    Window.iconphoto(False, PhotoImage(file='assets/Plot.png'))
    MainFrame = Frame(Window, bg='black', width=1000, height=1000, bd=0)
    MainFrame.pack(fill=BOTH, expand=True)
    HotbarFrame = Frame(Window, width=1000, height=100, bg='black', bd=0)
    HotbarFrame.pack(fill=BOTH, expand=True)
    PlayImage = PhotoImage(file='assets/play_button.png')
    PlayButton = Button(HotbarFrame, image=PlayImage, command=change_pause_status, relief=RAISED, bd=5, width=100, height=100)
    PlayButton.pack(fill=X)
    game_loop(Window, MainFrame, app=app)
    Window.mainloop()


def RightButtonF() -> None:
   global CHARACT
   CHARACT = 'clement'


def LeftButtonF() -> None:
    global CHARACT
    CHARACT = 'alex'


def ChooseCharacter() -> None:
    global CHARACT
    WindowCharacter = Tk()
    WindowCharacter.title("Attention sol mouillé")
    WindowCharacter.config(height=800, width=600, bd=2)
    WindowCharacter.resizable(False, False)
    WindowCharacter.iconphoto(False, PhotoImage(file='assets/Plot.png'))
    LeftFrame = Frame(WindowCharacter, height=800, width=300, bd=0)
    RightFrame = Frame(WindowCharacter, height=800, width=300, bd=0)
    LeftFrame.pack(side=LEFT, expand=True)
    RightFrame.pack(side=RIGHT, expand=True)
    LeftImage = PhotoImage(file='assets/SelectAlexSmall.png')
    RightImage = PhotoImage(file='assets/SelectClementSmall.png')
    LeftButtonImage = PhotoImage(file='assets/AlexButton.png')
    RightButtonImage = PhotoImage(file='assets/ClementButton.png')
    LeftSelection = Label(LeftFrame, image=LeftImage, bd=0, width=200, height=300)
    RightSelection = Label(RightFrame, image=RightImage, bd=0, width=200, height=300)
    LeftSelection.pack()
    RightSelection.pack()
    LeftButton = Button(LeftFrame, image=LeftButtonImage, bd=0, width=200, height=100, command=LeftButtonF)
    RightButton = Button(RightFrame, image=RightButtonImage, bd=0, width=200, height=100, command=RightButtonF)
    LeftButton.pack()
    RightButton.pack()
    while CHARACT is None:
        WindowCharacter.update()
        try:
            WindowCharacter.state()
        except tkinter.TclError:
            exit()
    WindowCharacter.destroy()
    WindowCharacter.mainloop()
    launch_game()


ChooseCharacter()
