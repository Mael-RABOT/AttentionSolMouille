import tkinter
from tkinter import *
from random import randint

ARR_SIZE = 10
EMPTY =  0
PLAYER = 1
ENEMY =  7
POC =   5
WATER = 3
MOVE_SET = [0, 1, 2, 3]
PAUSED = True
CHARACT = None
OBSTACLES_NUM = 10  # ~

def change_pause_status() -> None:
    global PAUSED
    PAUSED = not PAUSED

def spawn_player(Map: list[list[int]], PlayerPos: list[int, int]=None) -> list[int, int]:
    if PlayerPos is not None:
        Map[PlayerPos[0]][PlayerPos[1]] = EMPTY
    PlayerPos = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    Map[PlayerPos[0]][PlayerPos[1]] = PLAYER
    return PlayerPos

def spawn_poc(Map, PlayerPos, Obstacle):
    PoC = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    while PoC in Obstacle and PoC == PlayerPos:
        PoC = [randint(0, ARR_SIZE - 1), randint(0, ARR_SIZE - 1)]
    Map[PoC[0]][PoC[1]] = POC
    return PoC

def InputSelector() -> int:
    #Input = AI()
    Input = int(input("TO BE REPLACE BY AI SELECTION : "))
    while Input not in MOVE_SET:
        #Input = AI()
        Input = int(input("TO BE REPLACE BY AI SELECTION : "))
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

def ReversePoc(PoC) -> None:
    x = PoC[0]
    PoC[0] = PoC[1]
    PoC[1] = x

def game_loop(Window: tkinter.Tk, MainFrame: tkinter.Frame, enemy: bool=False, ObstacleNumber: int=OBSTACLES_NUM) -> None:
    Map = [[0 for j in range(ARR_SIZE)] for i in range(ARR_SIZE)]

    global PAUSED, CHARACT, KEY
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
    PoCImage = PhotoImage(file='assets/MiniPok.png')

    PlayerPos = spawn_player(Map)

    Obstacle = []

    WaterArr = [0 for i in range(ObstacleNumber + 1)]
    for i in range(ObstacleNumber):
        AddObstacle(Obstacle, PlayerPos)
        WaterArr[i] = MainCanvas.create_image(50, 50, image=WaterImage)
        MainCanvas.move(WaterArr[i], Obstacle[i][0] * 100, Obstacle[i][1] * 100)
    MiniPoc = MainCanvas.create_image(50, 50, image=PoCImage)

    Map[Obstacle[1][0]][Obstacle[1][1]] = WATER

    PoC = spawn_poc(Map, PlayerPos, Obstacle)
    MainCanvas.move(MiniPoc, PoC[0] * 100, PoC[1] * 100)

    while PlayerPos in Obstacle and PlayerPos == PoC:
        PlayerPos = spawn_player(Map,PlayerPos)

    MainCanvas.move(Player, PlayerPos[1] * 100, PlayerPos[0] * 100)

    def key_press(event):
        if event.char == 'z' or event.char == 'w':
            MovePlayer(Map, PlayerPos, 0, MainCanvas, Player)
        elif event.char == 's':
            MovePlayer(Map, PlayerPos, 1, MainCanvas, Player)
        elif event.char == 'q' or event.char == 'a':
            MovePlayer(Map, PlayerPos, 2, MainCanvas, Player)
        elif event.char == 'd':
            MovePlayer(Map, PlayerPos, 3, MainCanvas, Player)

    Window.bind('<KeyPress>', key_press)

    ReversePoc(PoC)

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
            if CheckPlot(Map, Obstacle, PlayerPos):
                exit()
            if PoC == PlayerPos:
                exit()
        if not SkipTurn:
            PlayerAction = InputSelector()
            MovePlayer(Map, PlayerPos, PlayerAction, MainCanvas, Player)
            if CheckPlot(Map, Obstacle, PlayerPos):
                exit()
            if PoC == PlayerPos:
                exit()

def launch_game() -> None:
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
    game_loop(Window, MainFrame)
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
