import tensorflow as tf

keras_model = tf.keras.models.load_model('A_NN7-2023-09-01_12-10-27-2023-09-01_13-42-35.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open('A_NN7-2023-09-01_12-10-27-2023-09-01_13-42-35.tflite', 'wb') as f:
    f.write(tflite_model)