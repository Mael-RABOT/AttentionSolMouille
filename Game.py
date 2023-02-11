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

import application
import microphone

ARR_SIZE = 10
EMPTY =  0
PLAYER = 1
ENEMY =  7
WALL =   5
MOVE_SET = [0, 1, 2, 3]
MOVE_SET_DICT = {"30" : MOVE_SET[0], "5" : MOVE_SET[1], "15" : MOVE_SET[2], "22" : MOVE_SET[0]}
ENTITY_SET = {"player": PLAYER, "enemy": ENEMY}
LIVES = 3
PAUSED = True

MAIN_FRAME_COLOR = '#000000'
HOTBAR_COLOR =     '#000000'

def change_pause_status() -> None:
    global PAUSED
    PAUSED = not PAUSED

def display_map(Map: list[int, int], clear: bool=True) -> None:
    if clear:
        subprocess.run("clear")
    for i in Map:
        print(i)

def spawn_player(Map: list[int, int]) -> list[int, int]:
    pos = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    Map[pos[0]][pos[1]] = PLAYER
    return pos

def spawn_enemy(Map: list[int, int], player_pos: list[int, int]) -> list[int, int]:
    enemy_pos = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    while enemy_pos == player_pos:
        enemy_pos = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    Map[enemy_pos[0]][enemy_pos[1]] = ENEMY
    return enemy_pos

def change_pos(Map: list[int, int], pos: list[int, int], action: int, entity_type: str) -> bool:
    if entity_type not in ENTITY_SET.keys():
        raise "Bad entity type, should be 'player' or 'enemy'"
    pos_backup = [pos[i] for i in range(len(pos))]
    Map[pos[0]][pos[1]] = EMPTY
    if action == 0:
        pos[0] = pos[0] - 1
    elif action == 1:
        pos[0] = pos[0] + 1
    elif action == 2:
        pos[1] = pos[1] - 1
    elif action == 3:
        pos[1] = pos[1] + 1
    else:
        return False
    if pos[0] >= ARR_SIZE or pos[1] >= ARR_SIZE or pos[0] < 0 or pos[1] < 0 or (entity_type == "player" and Map[pos[0]][pos[1]] == ENEMY) or (entity_type == "enemy" and Map[pos[0]][pos[1]] == PLAYER):
        if action == 0:
            pos[0] = pos[0] + 1
        elif action == 1:
            pos[0] = pos[0] - 1
        elif action == 2:
            pos[1] = pos[1] + 1
        elif action == 3:
            pos[1] = pos[1] - 1
        Map[pos[0]][pos[1]] = ENTITY_SET[entity_type]
        return False
    Map[pos[0]][pos[1]] = ENTITY_SET[entity_type]
    return True

def check_enemy_proximity(Map: list[int, int], player_pos: list[int, int], enemy_pos: list[int, int]) -> bool:
    if (Map[player_pos[0]][player_pos[1] + 1] == ENEMY) or (Map[player_pos[0]][player_pos[1] - 1] == ENEMY) or (Map[player_pos[0] + 1][player_pos[1]] == ENEMY) or (Map[player_pos[0] - 1][player_pos[1]] == ENEMY) or (Map[player_pos[0] + 1][player_pos[1] + 1] == ENEMY) or (Map[player_pos[0] + 1][player_pos[1] - 1] == ENEMY) or (Map[player_pos[0] - 1][player_pos[1] + 1] == ENEMY) or (Map[player_pos[0] - 1][player_pos[1] - 1] == ENEMY):
        return True
    return False

def fight(LIVES: int) -> None:
    LIVES -= 1

def InputSelector(Window, app) -> int:
    Input = None
    while Input not in MOVE_SET:
        Window.update()
        microphone.start_record(1)
        Input = app.execute_predict("./assets/file1.wav")
        Input = MOVE_SET_DICT[Input]
    return Input

def MovePlayer(Map: list[int, int], PlayerPos: list[int, int], PlayerAction: int, MainCanvas: tkinter.Canvas, Player: any) -> bool:
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


def game_loop(Window, MainFrame, Map, enemy=False, app=None):
    global PAUSED, LIVES
    Game = True

    MainCanvas = Canvas(MainFrame, bg='blue', width=1000, height=1000, bd=0)
    MainCanvas.pack(fill=BOTH, expand=True, side='top')
    BackgroundImage = PhotoImage(file='assets/DungeonButNotUgly.png')
    MainCanvas.create_image(500, 500, image=BackgroundImage, anchor=CENTER)
    PlayerImage = PhotoImage(file='assets/alex.png')
    Player = MainCanvas.create_image(50, 50, image=PlayerImage)

    PlayerPos = spawn_player(Map)
    if enemy:
        EnemyPos = spawn_enemy(Map, player_pos)

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
            PlayerAction = InputSelector(Window, app)
            if not MovePlayer(Map, PlayerPos, PlayerAction, MainCanvas, Player):
                LIVES -= 1
                SkipTurn = True

        Window.update()
        if LIVES <= 0:
            game = False

#MainCanvas.move(Player, 0.001, 0.001)

def gloop(Map: list[int, int], LIVES: int=3, enemy: bool=True, show_in_term: bool=False) -> None:
    global PAUSED
    game = True

    player_pos = spawn_player(Map)
    if enemy:
        enemy_pos = spawn_enemy(Map, player_pos)

    if show_in_term:
        display_map(Map)
        print(f"LIVES : {LIVES}")
    while game:
        while PAUSED:
            pass
        skip_turn = False
        try:
            if show_in_term:
                player_action = int(input(f"\n{MOVEMENTS}\n > "))
            else:
                player_action = receive_input()
        except ValueError:
            skip_turn = True
        if not skip_turn:
            if player_action in MOVE_SET.keys():
                if not change_pos(Map, pos=player_pos, action=player_action, entity_type="player"):
                    LIVES -= 1
                    skip_turn = True
        if not skip_turn:
            if enemy:
                enemy_action = randint(0, 4)
                change_pos(Map, pos=enemy_pos, action=enemy_action, entity_type="enemy")
                if check_enemy_proximity(Map, player_pos, enemy_pos):
                    fight(LIVES)
        if show_in_term:
            display_map(Map, clear=True)
            print(f"LIVES : {LIVES}")
        if LIVES <= 0:
            game = False
            if show_in_term:
                subprocess.run("clear")
                print("GAME OVER")

def launch_game(Map: list[int, int]) -> None:
    try:
        app = application.Application(model_path=sys.argv[1])
        app.load_trainloader()
        app.load_model()
    except IndexError:
        app = application.Application()
        app.load_trainloader()
        app.train_model()
    print(app.labels.index("up"))
    print(app.labels.index("down"))
    print(app.labels.index("left"))
    print(app.labels.index("right"))
    Window = Tk()
    Window.title("Attention sol mouillé")
    Window.geometry("1000x1100")
    Window.iconphoto(False, PhotoImage(file='assets/attention.png'))
    MainFrame = Frame(Window, bg=MAIN_FRAME_COLOR, width=1000, height=1000, bd=0)
    MainFrame.pack(fill=BOTH, expand=True)
    HotbarFrame = Frame(Window, width=1000, height=100, bg=HOTBAR_COLOR, bd=0)
    HotbarFrame.pack(fill=BOTH, expand=True)
    PlayImage = PhotoImage(file='assets/play_button.png')
    PlayButton = Button(HotbarFrame, image=PlayImage, command=change_pause_status, relief=RAISED, bd=5, width=100, height=100)
    PlayButton.pack(fill=X)
    game_loop(Window=Window, MainFrame=MainFrame, Map=Map, app=app)
    Window.mainloop()


print(sys.argv[1])
Map = [[0 for j in range(ARR_SIZE)] for i in range(ARR_SIZE)]
launch_game(Map)