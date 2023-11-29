# import datetime as datetime
# timestamp1 = datetime.datetime.now()#.strftime('%Y-%m-%d_%H-%M-%S')

# import numpy as np
# from PIL import Image
import tensorflow as tf
import cv2
# import requests
# import os

# interpreter = tf.lite.Interpreter(model_path='tflite/NN7-2023-08-31_04-18-35-2023-08-31_05-40-04.tflite')
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

#cap = cv2.VideoCapture(0)
classes = ['Frente+Umida', 'Fundo+Intermediaria', 'Meio+Seca', 'camendoim6sCenL1'] 


#while True:
#     _, frame = cap.read()
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#     cv2.imshow('img1', frame) #display the captured image
#     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 

#         img  = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_AREA)
#         name = 'images/camas'+timestamp+'.jpg'
#         cv2.imwrite(name, img)

#         image_content = img.resize((150, 150))
#         image_content = np.expand_dims(image_content, 0)
#         image_content = np.array(image_content, dtype=np.float32) / 255.0
#         interpreter.set_tensor(input_details[0]['index'], image_content)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         result = classes[output_data.argmax()]
#         print(result)

#      if cv2.waitKey(1) & 0xFF == ord('q'): #leave on pressing 'q' 
#          cv2.destroyAllWindows()
#          break

# cap.release()
# cv2.destroyAllWindows()








# image_directory = '2_frenteumida/'
# class_name = 'Frente+Umida'
# count = 0
# results = []

# for filename in os.listdir(image_directory):
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(image_directory, filename)

#         image_content =  Image.open(img_path) #plt.imread("images/camas.jpg")

#         image_content = image_content.resize((150, 150))

#         image_content = np.expand_dims(image_content, 0)

#         image_content = np.array(image_content, dtype=np.float32) / 255.0

#         interpreter.set_tensor(input_details[0]['index'], image_content)
#         interpreter.invoke()

#         output_data = interpreter.get_tensor(output_details[0]['index'])

#         classes = ['Frente+Umida', 'Fundo+Intermediaria', 'Meio+Seca', 'camendoim6sCenL1'] 

#         result = classes[output_data.argmax()]
#         results.append(result)

#         if result == class_name:
#             count += 1

# print(count, len(results), count/len(results))  
# print(results)  

       
# timestamp = datetime.datetime.now()#.strftime('%Y-%m-%d_%H-%M-%S')

# s3_bucket_url = f"https://rmfymrl340.execute-api.sa-east-1.amazonaws.com/dev/jetson-nano-images/{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

# headers = {"Content-Type": "image/jpeg"}

# with open(img_name, "rb") as f:
#             image = f.read()

# response = requests.put(s3_bucket_url, data=image, headers=headers)

#print(response, classes[output_data.argmax()], (timestamp - timestamp1).total_seconds())