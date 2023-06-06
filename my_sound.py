import tkinter as tk
import pygame


pygame.init()
pygame.mixer.init()
def play():
    print("Play")
    
    pygame.mixer.music.load("/home/vip/Documents/PyQt_Demo/Sound_2.mp3")
    pygame.mixer.music.play()



root = tk.Tk()

play_button = tk.Button(root, text='play',command=play)
play_button.pack()

root.mainloop()