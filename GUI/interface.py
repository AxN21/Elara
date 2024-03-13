import os.path
import pygame
import playsound
import time


# Interface for Elara using pygame
class imageHandler:
    def __init__(self) -> None:
        self.pics = dict()

    def loadFromFile(self, filename, id=None):
        if id == None: id = filename
        self.pics[id] = pygame.image.load(filename).convert()


    def loadFromSurface(self, surface, id):
        self.pics[id] = surface.convert_alpha()

    def render(self, surface, id, position=None, clear=False, size=None):
        if clear == True:
            surface.fill((5,2,23)) 
        if position is None:
            picX = int(surface.get_width() / 2 - self.pics[id].get_width() / 2)
            picY = int(surface.get_height() / 2 - self.pics[id].get_height() / 2)
        else:
            picX, picY = position

        if size is None:
            surface.blit(self.pics[id], (picX, picY))
        else:
            original_width, original_height = self.pics[id].get_size()
            aspect_ratio = original_width / original_height
             
            # Calculate new width based on the desired height to maintain aspect ratio
            new_width = int(size[1] * aspect_ratio)
            scaled_image = pygame.transform.smoothscale(self.pics[id], (new_width, size[1]))
            surface.blit(scaled_image, (picX, picY))


# Initialises the display
pygame.display.init()
pygame.display.set_caption("Elara")
screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
handler = imageHandler()


def display_images():
    # Normal Orb : Non-Speaking
    for i in range(1, 91):
        filename = f"/home/axn/Desktop/Python projects/Elara/elara_frames/{i}.jpg"
        handler.loadFromFile(filename, str(i))

    A = 0
    B = 0
    x = 400
    y = 400
    
    animation_clock = pygame.time.Clock()

    for i in range(1, 91):
        handler.render(screen, str(i), (A, B), True, (x, y))
        pygame.display.update(A, B, x, y)
        animation_clock.tick(30)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    display_images()


