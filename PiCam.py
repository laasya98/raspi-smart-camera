import RPi.GPIO as GPIO
import sys
import os
import numpy as np
import time
import pygame
from pygame.locals import*  # for event MOUSE variables
from collections import deque
import math
import io
import picamera
from picamera import PiCamera
import cv2
from scipy.interpolate import UnivariateSpline


# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.putenv('SDL_FBDEV', '/dev/fb1')
# os.putenv('SDL_MOUSEDRV', 'TSLIB') # Track Mouse clicks on piTFT
# os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')

####### INITIALIZATION ####################################

GPIO.setmode(GPIO.BCM)

# piTFT buttons
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pygame.init()
pygame.mouse.set_visible(True)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

menu_font = pygame.font.SysFont("caveat", 30)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen.fill(BLACK)

#####################Class Definition#####################


class Wheesh:
    def __init__(self):
        self.camera = PiCamera()
        # we can make this the same as ScreenWidth/Height if u want, or have the image take up a different size
        self.camera.resolution = (320, 240)

        self.menu_font = pygame.font.SysFont("caveat", 30)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen.fill(BLACK)
        self.stream = io.BytesIO

        # state system
        self._mainState = 0
        # screen dimensions x 3 channels
        self.rgb = bytearray(320 * 240 * 3)
        self.current_image = []
        self.edited_image = self.current_image
        self.n = 0


        # 0:free view, 1:captured picture display (show orignal), 2: edited image
        # 3:menu
        # jk, 3:holds all the menus
        # 8:Cloud


        # adjustment parameters
        self.alpha = 0
        self.blur = 0

    def inc(self):
        self.n += 1

    def CurrMode(self):
        return self._mainState

    def EnterState0(self):
        self._mainState = 0

    def EnterState1(self):
        self._mainState = 1

    def EnterState2(self):
        self._mainState = 2

    def EnterState3(self):
        self._mainState = 3

    ####### IMAGE PROCESSING ####################################

    def capture(self, rgb, save=False, n=0):
        stream = io.BytesIO()
        self.camera.capture(stream, resize=(320, 240),
                            use_video_port=True, format='rgb')
        stream.seek(0)
        stream.readinto(self.rgb)

        if save:
            self.camera.capture("img"+str(self.n)+".jpg")
            self.inc()
            stream.close()

            # decode = cv2.imdecode(np.asarray(rgb, np.uint8), cv2.IMREAD_COLOR)
            pgi = pygame.image.frombuffer(rgb, (320, 240), 'RGB')
            pgi_surf = pygame.surfarray.array3d(pgi)
            self.current_image = cv2.cvtColor(
                pgi_surf.transpose([1, 0, 2]), cv2.COLOR_RGB2BGR)
            self.edited_image = self.current_image

    def pygamify(self, image):
        # Convert cvimage into a pygame image
        # TODO: make this not a try catch
        try:
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return pygame.image.frombuffer(image2.tostring(), image2.shape[1::-1], "RGB")

    # Taken from building instagram-like filters in python
    def sepia(self, image):
        print "sepia"
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.filter2D(image, -1, kernel)

    def spreadLookupTable(self, x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    def warm_image(self, image):
        print "warm"
        increaseLookupTable = self.spreadLookupTable(
            [0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spreadLookupTable(
            [0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(image)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))

    def cold_image(self, image):
        print "cold"
        increaseLookupTable = self.spreadLookupTable(
            [0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.spreadLookupTable(
            [0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(image)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))

    def gray(self, image):
        print "gray"
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ####### SCREEN UPDATES ####################################

    def blit_text(self, s, pos):
        text = s
        text_surface = self.menu_font.render(s, True, BLACK)
        rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, rect)

    def blit_image(self, img, pos):
        pgi = self.pygamify(img)
        self.screen.blit(pgi, pos)
        pygame.display.update()

    # TODO: blit the little icons to make the display pretty on menu
    def blit_icon(self, img_path, pos):
        print "unimplemented"

    def blit_main_menu(self):
        print "hi"
        self.screen.fill(WHITE)
        self.blit_text("filter", (260, 60))
        self.blit_text("adjust", (80, 60))
        self.blit_text("art", (80, 180))
        self.blit_text("ML", (260, 180))
        pygame.display.update()

    def blit_adjust_menu(self):
        self.screen.fill(WHITE)
        self.blit_text("blur", (260, 60))
        self.blit_text("contrast", (80, 60))
        self.blit_text("brightness", (80, 180))
        self.blit_text("saturation", (260, 180))
        pygame.display.update()

    def blit_filter_menu(self):
        self.screen.fill(WHITE)
        self.blit_text("sepia", (260, 60))
        self.blit_text("noir", (80, 60))
        self.blit_text("warm", (80, 180))
        self.blit_text("cool", (260, 180))
        pygame.display.update()

    def blit_art_menu(self):
        self.screen.fill(WHITE)
        self.blit_text("edge", (260, 60))
        self.blit_text("8-bit", (80, 60))
        self.blit_text("vectorized", (80, 180))
        self.blit_text("free", (260, 180))
        pygame.display.update()

    def blit_ml_menu(self):
        self.screen.fill(WHITE)
        self.blit_text("emotion detection", (120, 60))
        self.blit_text("common object detection", (120, 180))
        pygame.display.update()
    
    def blit_adjust_bar(self):
        self.blit_text("emotion detection", (120, 60))
        self.blit_text("common object detection", (120, 180))
        pygame.display.update()

    ####### EVENT HANDLING ####################################

    def get_quadrant(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif(event.type is MOUSEBUTTONDOWN):
                pos = pygame.mouse.get_pos()
            elif(event.type is MOUSEBUTTONUP):
                pos = pygame.mouse.get_pos()
                x, y = pos
                # quit button (before game)
                if x > 180 and y > 120:
                    return 1
                elif x > 180 and y < 120:
                    return 2
                elif x < 180 and y > 120:
                    return 3
                else:
                    return 4
        return 0

    def handle_filter_menu(self, image):
        quad = self.get_quadrant()
        if quad == 4:
            print "gray"
            edited_image = self.gray(image)
            return edited_image, False
        elif quad == 2:
            print "sepia"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            edited_image = self.sepia(image)
            return edited_image, False
        elif quad == 1:
            print "warm"
            edited_image = self.warm_image(image)
            return edited_image, False
        elif quad == 3:
            print "cool"
            edited_image = self.cold_image(image)
            return edited_image, False
        return image, True

    def handle_main_menu(self, image):
        # case switch for each of the different quadrants
        quad = self.get_quadrant()
        if quad == 4:
            # open adjustment menu
            self.blit_adjust_menu()
            return image, False

        elif quad == 2:
            # open filtering l2 menu
            filtering = True
            self.blit_filter_menu()
            while filtering:
                image, filtering = self.handle_filter_menu(image)
            return image, False

        elif quad == 3:
            self.blit_art_menu()
            return image, False

        elif quad == 1:
            self.blit_ml_menu()
            return image, False

        return image, True


####### MAIN LOOP ####################################
w = Wheesh()

try:
    while True:
        # free view mode: menu isnt open and we aren't on a frame
        if w.CurrMode() == 0:  # free viewing mode, have ability to take a picture
            w.capture(w.rgb)
            img = pygame.image.frombuffer(w.rgb, (320, 240), 'RGB')
            w.screen.blit(img, (0, 0))
            # take a picture
            if (not GPIO.input(17)):
                w.capture(w.rgb, True, w.n)
                w.inc()
                print("picture taken")
                w.EnterState1()

        # captured picture display / (show orignal)
        if w.CurrMode() == 1:
            w.blit_image(w.current_image, (0,0)) #either update display right here, or move the blit into the "enter" functions
            if ( not GPIO.input(23) ):
                print "displaying edited image"
                w.EnterState2()

            # only open menu when frozen
            if ( not GPIO.input(22) ):
                print "opening main menu..."
                w.EnterState3()

            if (not GPIO.input(17)):
                # todo: open save menu
                w.EnterState0()


        # edited picture display (show edited)
        if w.CurrMode() == 2:
            w.blit_image(w.edited_image, (0,0))
            if ( not GPIO.input(23) ):
                print "displaying original image"
                w.EnterState1()

            # only open menu when frozen: can open menu from edited image
            if ( not GPIO.input(22) ):
                print "opening main menu..."
                w.EnterState3()

            if (not GPIO.input(17)):
                # todo: open save menu
                w.EnterState0()


        if w.CurrMode() == 3:
            # if main menu is not open
            w.blit_main_menu()
            main_menu_open = True
            # process menu actions:
            while main_menu_open:
                w.edited_image, main_menu_open = w.handle_main_menu(w.current_image)
            w.EnterState2()

        # quit at any time
        if ( not GPIO.input(27) ):
            print "Thanks for trying out the SmartCam :)"
            GPIO.cleanup()
            w.camera.close()
            quit()

        pygame.display.update()



except KeyboardInterrupt:
    GPIO.cleanup()
    w.camera.close()
    quit()

