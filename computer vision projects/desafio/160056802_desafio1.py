# C�digo feito por Vitor Moraes Dellamora

import cv2
import numpy as np

blue = 0
green = 0
red = 0

range_h = 3
range_s = 100
range_v = 180

target_h = 127#127
target_s = 100
target_v = 200

def click_event(event, x , y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		global target_h
		global target_s
		global target_v
		target_h = hsv[y, x, 0]
		target_s = hsv[y, x, 1]
		target_v = hsv[y, x, 2]
		
		print(str(target_h), ', ', str(target_s), ', ', str(target_v))

def within_range(value, target, margem):
	# margem = 5
	return value >= target-margem and value <= target+margem


#início
cap = cv2.VideoCapture(0) # acessa a câmera
fundo = cv2.imread('test.jpg') #carrega imagem que será utilizada no fundo
fundo = cv2.resize(fundo, (640, 480)) # redimensiona pro tam dos frames do vídeo

#loop de execução do vídeo da webcam
while(cap.isOpened()):
    ret, frame = cap.read() # pega o frame da vez
    
    kernel = np.ones((3,3),np.float32)/25
    dst = cv2.filter2D(frame,-1,kernel)
    
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)# converte pra HSV

    lower_green = np.array([target_h - range_h, target_s - range_s, target_v - range_v])
    upper_green = np.array([target_h + range_h, target_s + range_s, target_v + range_v])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)


	# bg = cv2.bitwise_and(frame, frame, mask=mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    produto_final = np.where(fg==0, fundo, frame)
    cv2.imshow("160056802", produto_final)
    aBGR = cv2.setMouseCallback('160056802', click_event)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        print(str(blue), ', ', str(green), ', ', str(red))
        break

cap.release()
cv2.destroyAllWindows()