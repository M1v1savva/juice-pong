import tkinter as tk
from tkinter.constants import INSERT
import cv2
from PIL import Image, ImageTk, ImageDraw

width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height/2)

root = tk.Tk()
text = tk.Label(root, font=15)
root.title("Ball Throwing Robot")
initial_player = 0 # 0 for robot, 1 for human
if initial_player == 0:
    text['text'] = "The robot is throwing"
    text['fg'] = '#f00'
elif initial_player == 1:
    text['text'] = "The human is throwing"
    text['fg'] = '#00f'
text.pack()

root.attributes("-fullscreen", True)
root.bind('<F11>', lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind('<Escape>', lambda e: root.quit())
root.bind('<F2>', lambda e: switch_player(text))
can = tk.Canvas(root)
lmain = tk.Label(root)
can.pack()
lmain.pack()

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
    if text['text'] == "The robot is throwing":
        text['text'] = "The human is throwing"
        text['fg'] = '#00f'
    elif text['text'] == "The human is throwing":
        text['text'] = "The robot is throwing"
        text['fg'] = '#f00'
    return

show_frame()
root.mainloop()