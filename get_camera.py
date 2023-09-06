import cv2
import datetime
import time

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

capture_interval = 30
last_capture_time = time.time()

while True:
    _, frame0 = cap0.read()
    _, frame1 = cap1.read()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow('Camera 0', frame0)
    cv2.imshow('Camera 1', frame1)

    key = cv2.waitKey(1)

    current_time = time.time()

    if key & 0xFF == ord('y') or (current_time - last_capture_time) >= capture_interval:
        img0 = cv2.resize(frame0, (300, 300), interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(frame1, (300, 300), interpolation=cv2.INTER_AREA)

        name0 = 'images/camera0_' + timestamp + '.jpg'
        name1 = 'images/camera1_' + timestamp + '.jpg'

        cv2.imwrite(name0, img0)
        cv2.imwrite(name1, img1)

        last_capture_time = current_time

    elif key & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()













# -----------------------------------------------
# import cv2
# import datetime as datetime
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#     cv2.imshow('img1', frame) #display the captured image
#     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
#         img  = cv2.resize(frame, (128, 128), interpolation = cv2.INTER_AREA)
#         name = 'images/camas'+timestamp+'.jpg'
#         cv2.imwrite(name, img)
        
#     if cv2.waitKey(1) & 0xFF == ord('q'): #leave on pressing 'q' 
#         cv2.destroyAllWindows()
#         break

# cap.release()
# cv2.destroyAllWindows()