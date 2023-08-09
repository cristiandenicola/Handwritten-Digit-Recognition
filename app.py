import pygame, sys
from pygame.locals import *
from pygame import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import cv2

## Costanti dimensione window
WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480

# Dim boundry del rettangolo
BOUNDRYINC = 5

# Colori
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0 ,0)

# Caricamento del modello di intelligenza
MODEL = load_model("bestAcc_model.h5")

# Dizionario di associazione per le predizioni del modello
LABELS = {0:"Zero", 1:"One", 
        2:"Two", 3:"Three",
        4:"Four", 5:"Five",
        6:"Six", 7:"Seven",
        8:"Eight", 9:"Nine"}

# Inizializzazione pygame
pygame.init()
pygame.display.set_caption("Digit recognition")

FONT = pygame.font.Font("freesansbold.ttf", 18)

# Superficie disponiibile per l'utente
DISPLAYSURF = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))

#variabili globali del while
iswriting = False

# Coordinate che salvano la posizione dove l'utente ha scritto
number_xcord = []
number_ycord = []

# Inizializzazione dati di addestramento
X_train = np.empty((0, 28, 28, 1), dtype=np.float32)
y_train = np.array([], dtype=np.int32)

# Predizione del modello
PREDICT = True


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # Calcolo dim x e y del rettangolo in base a dove si è triggerato l'evento
            rect_min_x, rect_max_x = min(number_xcord) - BOUNDRYINC, max(number_xcord) + BOUNDRYINC
            rect_min_y, rect_max_y = min(number_ycord) - BOUNDRYINC, max(number_ycord) + BOUNDRYINC

            # Disegna il rettangolo sulla superficie DISPLAYSURF
            pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

            # Reset delle coordinate
            number_xcord = []
            number_ycord = []
                
            # Creo ROI (area di interesse rettangolare)
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
                
            ## Inizio previsione del modello
            if PREDICT:
                # Standardizzazione dell'image per consentire al modello di leggerla
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values = 0)
                image = cv2.resize(image, (28, 28))/255

                # Label contenente la previsione del modello sull'input
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)
            
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)
        
        pygame.display.update()

