import RPi.GPIO as GPIO
import time
import cv2
import datetime

# configura entradas
GPIO.setmode(GPIO.BCM)
GPIO.setup(9, GPIO.IN)

# configura intervalo de captura
capture_interval = 1800 # 1800 segundos = 30 minutos
last_capture_time = time.time()

while(True):
    i = GPIO.input(9)
    print("input status:", i)
    
    # horário atual
    current_time = time.time()
    
    # se o input = 1 e última captura >= 30 minutos
    if i == 1 and ((current_time - last_capture_time) >= capture_interval):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # configura câmeras
        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(1)
        
        # erro em alguma das câmeras
        if not cap0.isOpened() or not cap1.isOpened():
            print("Error: Cameras not found")
            break

        # lê captura imagens das câmeras
        _, frame0 = cap0.read()
        _, frame1 = cap1.read()

        # rotate
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180) # camera termica
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE) # camera rgb

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
        
        # atualiza última captura
        last_capture_time = current_time
            
    # 1 segundo entre cada leitura do input digital
    time.sleep(1)

# libera recursos usados pelo opencv
cv2.destroyAllWindows()

        
