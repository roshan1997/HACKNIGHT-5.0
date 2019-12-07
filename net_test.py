# import the necessary packages	
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2



# load the image
image1 = cv2.imread('testt.jpeg')
orig = image1.copy()

# pre-process the image for classification
#image = cv2.resize(image, (28, 28))
image = load_img('testt.jpeg',target_size=(28,28))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('model_new1.h5')

# classify the input image
(normal, pnemonia) = model.predict(image)[0]


# build the label
label = "NORMAL" if normal > pnemonia else "PNEMONIA"
proba = normal if normal > pnemonia else pnemonia
#label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)