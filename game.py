import cv2
import numpy as np
from collections import deque


class ClassName:

    def __init__(self, height, width,scale, maxlen=2 ):
        self.height = height//scale
        self.width = width//scale
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.background = None
        self.scale =scale

    def update_frame(self,frame):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(frame)
            self.bg1()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.bg2(old_frame, frame)

    def bg1(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for i in self.buffer:
            self.background += i
        self.background //= len(self.buffer)

    def bg2(self,old_frame,new_frame):
        self.background -= old_frame/ self.maxlen
        self.background += new_frame/ self.maxlen

    def get_bg(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        downscale = cv2.resize(frame, (self.width, self.height))
        grey = cv2.cvtColor(downscale, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        self.update_frame(grey)
        grey = cv2.absdiff(bg_buffer.get_bg(), grey)
        _, mask = cv2.threshold(grey, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (self.width*self.scale, self.height*self.scale))
        return mask


class Game:
    def __init__(self, height, width, size=100):
        self.height = height
        self.width = width
        self.size = size
        self.logo = cv2.imread("ghost.jpg")
        self.logo = cv2.resize(self.logo, (self.size, self.size))
        grey = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)[1]
        self.y= 0
        self.x= np.random.randint(0, self.width - self.size)
        self.speed = 5
        self.score = 0

    def update_frame(self, frame):
        roi = frame[self.y: self.y+self.size, self.x: self.x+self.size]
        roi[np.where(self.mask)] = 0
        roi += self.logo

    def update_position(self, mask):
        print(self.y)

        if self.y+self.size >= self.width:
            self.score += 1
            self.y = 0
            self.x = np.random.randint(0, self.width-self.size)
        self.y += self.speed

        roi = mask[self.y: self.y+self.size, self.x: self.x+self.size]
        flag = np.any(roi[np.where(self.mask)])
        if flag:
            self.score -= 1
            self.y = 0
            self.x = np.random.randint(0, self.width-self.size)
        return flag



height = 640
width = 480
scale = 2

webcam = cv2.VideoCapture(0)

bg_buffer = ClassName(height, width,scale,5)
game = Game(height, width)

while True:
    _, frame = webcam.read()
    frame = cv2.resize(frame, (height,width))
    frame = cv2.flip(frame, 1)
    mask = bg_buffer.apply(frame)
    flag = game.update_position(mask)
    if flag:
        frame[: , :, ] = (255, 0, 0)
    game.update_frame(frame)

    text = f"SCORE: {game.score}"
    cv2.putText(frame, text, (50,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0),2 )
    cv2.imshow("Mask", mask)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break











