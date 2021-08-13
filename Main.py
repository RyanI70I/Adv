import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from keras import models
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM




batch_size = 32
img_height = 512
img_width = 512
data_dir = "/Users/ryankersten/Desktop/Data/"



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=.2,
    seed=123,
    image_size=(img_height, img_width),
    subset="training",
    batch_size=batch_size)

checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)


class_names = train_ds.class_names
print(class_names)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
num_classes = 3


model = models.Sequential(name="test_model")

model.add(Conv2D(name='Conv1', filters=16, kernel_size=(2,2), activation='relu', input_shape= (512,512,3)))

model.add(MaxPooling2D(name='Conv12',pool_size=(4,4), strides=(2,2)))

model.add(Conv2D(name='Conv13',filters=16, kernel_size=(4,4), activation='relu'))

model.add(MaxPooling2D(name='Conv14',pool_size=(4,4), strides=(2,2)))

model.add(Conv2D(name='Conv15',filters=16, kernel_size=(4,4), activation='relu' ))



model.add(MaxPooling2D(name='Conv16',pool_size=(4,4), strides=(2,2)))

model.add(Conv2D(name='Conv18',filters=16, kernel_size=(4,4), activation='relu' ))

model.add(MaxPooling2D(name='Conv19',pool_size=(4,4), strides=(2,2)))
 
model.add(Conv2D(name='Conv11',filters=16, kernel_size=(4,4), activation='relu' ))

model.add(MaxPooling2D(name='Conv21',pool_size=(4,4), strides=(2,2)))

model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation
='softmax'))

model.build(input_shape=(None,512,512,3))
model.compile(
    loss="categorical_crossentropy", optimizer="Adadelta", metrics=[tf.keras.metrics.Accuracy()])
print(model.summary())

epochs = 40
history = model.fit(
    train_ds,
    epochs=epochs
)
