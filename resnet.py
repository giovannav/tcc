# Imports
import os
import random
import sys
import argparse
import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

# set the number of classes
num_classes = 4

def load_model(model_name, num_output_nodes, num_epochs, img_shape, batch_size, learning_rate):
    
    resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(img_shape, img_shape, 3))

    for layer in resnet_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        resnet_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create data generators for train, test, and validation sets
    train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(
            'camas_tiles_train',
            target_size=(img_shape, img_shape),
            batch_size=batch_size,
            class_mode='categorical'
        )

    test_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(
            'camas_tiles_test',
            target_size=(img_shape, img_shape),
            batch_size=batch_size,
            class_mode='categorical'
        )

    val_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(
            'camas_tiles_validation',
            target_size=(img_shape, img_shape),
            batch_size=batch_size,
            class_mode='categorical'
        )

    # Set the number of steps per epoch
    train_steps = len(train_data_gen)
    test_steps = len(test_data_gen)
    val_steps = len(val_data_gen)

    early_stopping = EarlyStopping(patience=5)

    print(train_steps, test_steps, val_steps)

    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_steps,
        validation_data=val_data_gen,
        validation_steps=val_steps,
        epochs=num_epochs,
        callbacks=[early_stopping]
    )
    
    epoch_list = list(range(1, num_epochs + 1))
    accuracy_list = history.history['accuracy']
    loss_list = history.history['loss']
    val_accuracy_list = history.history['val_accuracy']
    val_loss_list = history.history['val_loss']
    
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data_gen, steps=test_steps)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'model-{model_name}-outputnodes-{num_output_nodes}-epochs-{num_epochs}-imgshape-{img_shape}-batchsize-{batch_size}-{timestamp}'
    model.save(f'results_h5/{model_name}.h5')

    # evaluating the model with recall, precision and f1 score
    predictions = []
    labels = []
    num_samples = 0
    
    class_names = test_data_gen.class_indices
    class_names = list(class_names.keys())
    
    desired_num_predictions = 364 #len(test_data_gen)

    for x, y in test_data_gen:
        batch_size = x.shape[0]
        batch_predictions = np.argmax(model.predict(x), axis=-1)

        predictions.extend(batch_predictions)
        labels.extend(np.argmax(y, axis=-1))
        num_samples += batch_size

        if num_samples >= desired_num_predictions:
            break

    conf_matrix = confusion_matrix(labels, predictions)

    predictions = np.array(predictions[:desired_num_predictions])
    labels = np.array(labels[:desired_num_predictions])

    class_recall = recall_score(labels, predictions, average=None)
    class_precision = precision_score(labels, predictions, average=None)
    class_f1_score = f1_score(labels, predictions, average=None)

    file_path = f"results_txt/{model_name}_evaluation.txt"
    
    with open(file_path, "w") as file:
        file.write(f'Test Loss: {test_loss:.4f} \n')
        file.write(f'Test Accuracy: {test_accuracy:.4f} \n')
        file.write(f'Conf matrix: {str(conf_matrix)} \n')
        file.write(f'Class names: {str(class_names)} \n')
        file.write(f'Class recall: {str(class_recall)} \n')
        file.write(f'Class precision {str(class_precision)} \n')
        file.write(f'Class f1 score {str(class_f1_score)} \n')
        file.write(f'Epoch list {str(epoch_list)} \n')
        file.write(f'Accuracy list {str(accuracy_list)} \n')
        file.write(f'Val Accuracy list {str(val_accuracy_list)} \n')
        file.write(f'Loss list {str(loss_list)} \n')
        file.write(f'Val Loss list {str(val_loss_list)} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model using VGG16 transfer learning.")
    parser.add_argument('--model', type=str, default='vgg16', help='Model name to use')
    parser.add_argument('--output_nodes', type=int, default=256, help='Number of output nodes for the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=64, help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()
    load_model(args.model, args.output_nodes, args.epochs, args.img_size, args.batch_size, args.learning_rate)
