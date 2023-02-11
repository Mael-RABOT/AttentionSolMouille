import tkinter

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
from random import randint
import subprocess

ARR_SIZE = 10
EMPTY =  0
PLAYER = 1
ENEMY =  7
WALL =   5
MOVEMENTS = "UP : 0, DOWN : 1, LEFT : 2, RIGHT : 3"
MOVE_SET = {0 : [-1, 0], 1 : [+1, 0], 2 : [-1, 0], 3 : [+1, 0]}
ENTITY_SET = {"player": PLAYER, "enemy": ENEMY}
LIVES = 3

def make_window() -> any:
    Window = Tk()
    Window.title("Attention sol mouillé")
    Window.geometry("1080x1080")
    Window.iconphoto(False, PhotoImage(file='assets/attention.png'))
    can = Canvas(Window, bg="#000000")
    can.pack(fill=Y)
    return Window

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

def gloop(Map: list[int, int], LIVES: int=3, enemy=True) -> None:
    game = True

    player_pos = spawn_player(Map)
    if enemy:
        enemy_pos = spawn_enemy(Map, player_pos)

    display_map(Map)
    print(f"LIVES : {LIVES}")
    while game:
        skip_turn = False
        try:
            player_action = int(input(f"\n{MOVEMENTS}\n > "))
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
        display_map(Map, clear=True)
        print(f"LIVES : {LIVES}")
        if LIVES <= 0:
            game = False
            subprocess.run("clear")
            print("GAME OVER")

def launch_game(Map: list[int, int], init_window: bool=True) -> None:
    if init_window:
        Window = make_window()

    gloop(Map,enemy=False)

    if init_window:
        Window.mainloop()

Map = [[0 for j in range(ARR_SIZE)] for i in range(ARR_SIZE)]
launch_game(Map, init_window=False)