from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras import backend as K

BS = 32
inputShape = (28,28, 3)
classes = 2

model = Sequential()
'''
model.add(Conv2D(filters=16, kernel_size=1, activation="relu", input_shape=(28,28,3), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=1, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
'''
model.add(Dense(784, activation='relu', input_shape=inputShape))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os

EPOCHS = 50
INIT_LR = 1e-3

print("loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images('chest_xray')))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	#print(label)
	label = 1 if label == "NORMAL" else 0
	#print(label)
	labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
labels, test_size=0.25, random_state=42)


trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

print("compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
metrics=["accuracy"])

print("training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
epochs=EPOCHS, verbose=1)

score = model.evaluate(testX,testY)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
print("serializing network...")

model.save('model_new1.h5')
model.save_weights('weights_new1.h5')
