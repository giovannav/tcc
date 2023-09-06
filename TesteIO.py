import RPi.GPIO as GPIO
import time
import cv2
import datetime

GPIO.setmode(GPIO.BCM)

GPIO.setup(9,GPIO.IN)

while(True):
    print("ler estado: ")
    i = GPIO.input(9)
    print(i)

    if i==1:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(1)

        _, frame0 = cap0.read()
        _, frame1 = cap1.read()

        rame0 = cv2.rotate(frame0, cv2.ROTATE_180) # camera termica
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE) # camera rgb

        img0 = cv2.resize(frame0, (300, 300), interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(frame1, (300, 300), interpolation=cv2.INTER_AREA)

        name0 = 'robot_images/camera0_' + timestamp + '.jpg'
        name1 = 'robot_images/camera1_' + timestamp + '.jpg'

        cv2.imwrite(name0, img0)
        cv2.imwrite(name1, img1)

        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()


    time.sleep(0.5)