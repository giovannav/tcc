# import requests
import datetime as datetime
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('results_h5/model-NN5-3-layers-256-128-64-epochs-100-imgshape-128-batchsize-8-2023-08-11_00-38-07.h5')

while True:
    print('enter key')
    x = int(input())
    if x == 1:
        print("------>")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        img_name = 'images/frentemeioseca/camas2023-08-25_16-28-27.jpg'

        with open(img_name, "rb") as f:
            image = f.read()

        image_content =  Image.open(img_name) #plt.imread("images/camas.jpg")
            
        image_content = (np.expand_dims(image_content, 0))
            
        predictions_single = model.predict(image_content)

        # classes = ['Frente+Umida', 'Fundo+Intermediaria', 'Meio+Seca', 'camendoim6sCenL1'] 
            
        # s3_bucket_url = f"https://rmfymrl340.execute-api.sa-east-1.amazonaws.com/dev/jetson-nano-images/{timestamp}.jpg"

        # headers = {
        #     "Content-Type": "image/jpeg"
        # }

        # response = requests.put(s3_bucket_url, data=image, headers=headers)

        # print(response, predictions_single, max(predictions_single.tolist()[0]))
        print(predictions_single, max(predictions_single.tolist()[0]))
        break