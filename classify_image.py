import requests
import datetime as datetime
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# device = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device[0], True)
# tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

with open("images/camas.jpg", "rb") as f:
     image = f.read()

image_content = plt.imread("images/camas.jpg")
    
image_content = (np.expand_dims(image_content, 0))
    
model = load_model('results_h5/model-NN5-3-layers-256-128-64-epochs-100-imgshape-128-batchsize-8-2023-08-11_00-38-07.h5')

predictions_single = model.predict(image_content)

classes = ['Frente+Umida', 'Fundo+Intermediaria', 'Meio+Seca', 'camendoim6sCenL1'] 
    
s3_bucket_url = f"https://rmfymrl340.execute-api.sa-east-1.amazonaws.com/dev/jetson-nano-images/{timestamp}.jpg"

headers = {
    "Content-Type": "image/jpeg"
}

response = requests.put(s3_bucket_url, data=image, headers=headers)

print(response, predictions_single)