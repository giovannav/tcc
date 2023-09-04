import os
import random
import sys
import argparse
import datetime

import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

# device = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device[0], True)
# tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

def build_model(input_shape, num_classes):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


    for layer in vgg_model.layers:  # Fine-tune the last few layers
        layer.trainable = False
        
    model = tf.keras.Sequential([
        vgg_model,
        tf.keras.layers.Flatten(),
        #tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'), 
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'), 
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
        
    # model = tf.keras.Sequential([
    #     vgg_model,
    #     tf.keras.layers.GlobalAveragePooling2D(),  # Use Global Average Pooling
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.3),  # Add dropout for regularization
    #     tf.keras.layers.Dense(num_classes, activation='softmax')
    # ])

    # model = tf.keras.Sequential([
    #     vgg_model,
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(256, activation='relu'), 
    #     tf.keras.layers.Dense(128, activation='relu'), 
    #     #tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(num_classes, activation='softmax')
    # ])

    return model

def train_model(num_epochs, img_shape, batch_size, learning_rate):
    num_classes = 4
    input_shape = (img_shape, img_shape, 3)
    
    model = build_model(input_shape, num_classes)
    
    train_data_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=180,
                zoom_range=0.2,
                brightness_range=[0, 2],
                horizontal_flip=True,
                vertical_flip=True
            ).flow_from_directory(
                'camas_tiles_train',
                target_size=(img_shape, img_shape),
                batch_size=batch_size,
                class_mode='categorical'
            )
            
    test_data_gen = ImageDataGenerator(
            rescale=1./255
            ).flow_from_directory(
                'camas_tiles_test',
                target_size=(img_shape, img_shape),
                batch_size=batch_size,
                class_mode='categorical'
            )
            
    val_data_gen = ImageDataGenerator(
            rescale=1./255
            ).flow_from_directory(
                'camas_tiles_validation',
                target_size=(img_shape, img_shape),
                batch_size=batch_size,
                class_mode='categorical'
            )
    
    timestamp_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.000001, verbose=1)
    csv_logger = CSVLogger(filename=f'results_csv/VGG16-{timestamp_start}.csv', separator=',', append=False)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_steps = len(train_data_gen)
    val_steps = len(val_data_gen)
    
    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_steps,
        validation_data=val_data_gen,
        validation_steps=val_steps,
        epochs=num_epochs,
        callbacks=[early_stopping, lr_scheduler, csv_logger]
    )

    return model, history, test_data_gen, timestamp_start, num_epochs

def evaluate_model(model, history, test_data_gen, timestamp_start, num_epochs):
    test_steps = len(test_data_gen)
    test_loss, test_accuracy = model.evaluate(test_data_gen, steps=test_steps)
    
    epoch_list = list(range(1, num_epochs + 1))
    accuracy_list = history.history['accuracy']
    loss_list = history.history['loss']
    val_accuracy_list = history.history['val_accuracy']
    val_loss_list = history.history['val_loss']
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'VGG16-{timestamp_start}-{timestamp}'
    model.save(f'results_h5/{model_name}.h5')

    predictions = []
    labels = []
    num_samples = 0
        
    class_names = test_data_gen.class_indices
    class_names = list(class_names.keys())
        
    desired_num_predictions = 309 #len(test_data_gen)

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
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=150, help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    args = parser.parse_args()
    
    model, history, test_data_gen, timestamp_start, num_epochs = train_model(args.epochs, args.img_size, args.batch_size, args.learning_rate)
    evaluate_model(model, history, test_data_gen, timestamp_start, num_epochs)

