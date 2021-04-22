import pygame
import time

def playMusic(music):
    #file loading
    #play and pause
    loadMusic(music)
    while True:
        # timer = pygame.mixer.music.get_pos()
        # time.sleep(1)
        control = input()
        # pygame.time.Clock().tick(10)
        if control == " ":
            pygame.mixer.music.pause()
        elif control == "s" :
            pygame.mixer.music.unpause()
        elif control == "r" :
            loadMusic(music)
        elif control == "q":
            break

def loadMusic(music):
    pygame.mixer.init()
    pygame.mixer.music.load(music)
    print("Playing:", music)
    pygame.mixer.music.play()

if __name__ == '__main__':
    music = 'music.wav'
    playMusic(music)