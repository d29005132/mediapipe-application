import numpy as np
import webbrowser
import cv2
import pygame
import mediapipe as mp
import time
def intersection(a,b):
      x = max(a[0], b[0])
      y = max(a[1], b[1])
      w = min(a[0]+a[2], b[0]+b[2]) - x
      h = min(a[1]+a[3], b[1]+b[3]) - y
      if w<0 or h<0: return ()
      return (x, y, w, h)

def putText(source, x, y, text, scale=2.5, color=(255,255,255)):
        org = (x,y)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = scale
        thickness = 5
        lineType = cv2.LINE_AA
        cv2.putText(source, text, org, fontFace, fontScale, color, thickness, lineType)
        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

def loop_func(func, second):
 #每隔second秒执行func函数
 while True:
  timer = Timer(second, func)
  timer.start()
  timer.join()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,60 )

Rect_1_count = 0
Rect_2_count = 0
Rect_3_count = 0
frame_count = 0
web_count = 0
a=0
previous_frame = None
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while (True):
    frame_count += 1
        
    success, img = cap.read()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    img=cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w = imgRGB.shape[1]
    h = imgRGB.shape[0]
    white = 255 - np.zeros((h,w,4), dtype='uint8')
   

    results = hands.process(imgRGB)
  
    #print(results.multi_ hand_landmarks)

    rx=cv2.rectangle(img, (40, 440), (340, 640), (0, 0, 0), 5)
    ry=cv2.rectangle(img, (940, 40), (1240, 240), (0, 0, 0), 5)
    # cv2.rectangle(img, (1040, 540), (1240, 640), (0, 0, 0), 5)
    
            # 左邊方框內文字顯示
    if Rect_1_count<=1:
        cv2.putText(img, 'open cv', (60, 500), cv2.FONT_HERSHEY_DUPLEX, 
                    1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'opencv:{}%'.format(int((Rect_1_count/50)*100)), 
                    (60, 500), cv2.FONT_HERSHEY_DUPLEX, 
                    1, (0, 0, 0), 1, cv2.LINE_AA)
    # 右邊方框內文字顯示
    if Rect_2_count<=1:
        cv2.putText(img, 'play music', (960, 100), cv2.FONT_HERSHEY_DUPLEX, 
                    1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'play music:{}%'.format(int((Rect_2_count/50)*100)), 
                    (960, 100), cv2.FONT_HERSHEY_DUPLEX, 
                    1, (0, 0, 0), 1, cv2.LINE_AA)

    # if Rect_3_count<=1:
    #     cv2.putText(img, 'exit', (1060, 500), cv2.FONT_HERSHEY_DUPLEX, 
    #                 1, (0, 0, 0), 1, cv2.LINE_AA)
    # else:
    #     cv2.putText(img, 'exit'.format(int((Rect_3_count/50)*100)), 
    #                 (1060, 500), cv2.FONT_HERSHEY_DUPLEX, 
    #                 1, (0, 0, 0), 1, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                x = handLms.landmark[7].x * w   # 取得食指末端 x 座標
                y = handLms.landmark[7].y * h   # 取得食指末端 y 座標
                print(x,y)
                if 1200>x>966 and 95<y<270  :
                  pygame.mixer.init()
                  pygame.mixer.music.load('light.mp3')
                  pygame.mixer.music.play()
                if 60<x<330 and 470<y<680: 
                  web_count+=1
                  if web_count>300:
                   webbrowser.open('https://opencv.org/')
                   web_count=0
                
                    
                 
                 
                 
                
                   
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
 
           
           
           
           
           
           
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
   
 
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) == ord('e'):
       exit()