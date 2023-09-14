import RPi.GPIO as GPIO
import time
import cv2
import datetime

# configura entradas
GPIO.setmode(GPIO.BCM)
GPIO.setup(9, GPIO.IN)

while(True):
    i = GPIO.input(9)
    print("input status:", i)
        
    # se o input = 1
    if i == 1:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # configura câmeras
        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(1)

        # lê captura imagens das câmeras
        _, frame0 = cap0.read()
        _, frame1 = cap1.read()

        # rotate
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180) # camera termica
        #frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE) # camera rgb

        # resize
        img0 = cv2.resize(frame0, (300, 300), interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(frame1, (300, 300), interpolation=cv2.INTER_AREA)

        # renomeia imagens
        name0 = 'robot_images/camera0_' + timestamp + '.jpg'
        name1 = 'robot_images/camera1_' + timestamp + '.jpg'

        # salva imagens
        cv2.imwrite(name0, img0)
        cv2.imwrite(name1, img1)

        # fecha conexão com a câmera
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        
        # tira uma foto a cada 1 segundo enquanto o sinal for positivo
        time.sleep(1)
            
    elif i == 0:
        print('sleeping')
        time.sleep(1800) # para verificação por meia hora
        
# libera recursos usados pelo opencv
cv2.destroyAllWindows()

        
