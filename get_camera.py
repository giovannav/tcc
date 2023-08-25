# import numpy as np
# import time
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import cv2
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.experimental.set_visible_devices([], 'GPU')

#model = load_model("results_h5/model-NN5-3-layers-256-128-64-epochs-100-imgshape-128-batchsize-8-2023-08-11_00-38-07.h5")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    # img  = cv2.resize(frame, (128, 128), interpolation = cv2.INTER_AREA)
    # img = (np.expand_dims(img,0))
    # predictions_single = model.predict(img)
    # classes = ['Frente+Umida', 'Fundo+Intermediaria', 'Meio+Seca', 'camendoim6sCenL1'] 
    # result = classes[predictions_single.argmax()]
    # print(result)
    
    cv2.imshow('img1', frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        img  = cv2.resize(frame, (128, 128), interpolation = cv2.INTER_AREA)
        cv2.imwrite('images/camas.jpg', img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): #leave on pressing 'q' 
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()