import tkinter as tk
import UserLogin
from tkinter import messagebox
from functools import partial
import cv2
from PIL import Image, ImageTk

width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def loginAction(entries):
    print(UserLogin.attemptSession(entries[0].get(),entries[1].get()))

global root 

root = tk.Tk()
entries = {}
#root.attributes("-fullscreen", True)
can = tk.Canvas(root)
can.pack()
lmain = tk.Label(root)

usernameWord = tk.Label(can,text="Username",font=("Arial",10))
usernameWord.pack()
username = tk.Entry(can,width=40,font=("Arial",10))
username.pack()
entries[0] = username
passwordWord = tk.Label(can,text="Password",font=("Arial",10))
passwordWord.pack()
password = tk.Entry(can,width=40,show="\u25CF",font=("Arial",10))
password.pack()
entries[1] = password
trylogin = partial(loginAction,entries)
loginButton = tk.Button(can,text="Login",height=1,width=10,command=trylogin,font=("Arial",10))
loginButton.pack()
text = tk.Label(can, pady=50, font=("Arial",25))
root.title("Ball Throwing Robot")
lmain.pack()

initial_player = 0 # 0 for robot, 1 for human

if initial_player == 0:
    text['text'] = "The robot is throwing."
    text['fg'] = '#f00'
elif initial_player == 1:
    text['text'] = "The human is throwing."
    text['fg'] = '#00f'
text.pack()

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

def switch_player(text):
    if text['text'] == "The robot is throwing.":
        text['text'] = "The human is throwing."
        text['fg'] = '#00f'
    elif text['text'] == "The human is throwing.":
        text['text'] = "The robot is throwing."
        text['fg'] = '#f00'
    return

root.bind('<F11>', lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind('<Escape>', lambda e: root.quit())
root.bind('<F2>', lambda e: switch_player(text))
root.bind('<F1>', lambda e: tk.controller.show_frame("frame2"))

show_frame()
root.mainloop()
