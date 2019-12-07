from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
from keras import backend as K
import numpy as np
import argparse
import imutils
import cv2
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('test.html')

@app.route('/submit', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      file = secure_filename(f.filename)
      f.save(secure_filename(file))
      return redirect(url_for('predict',filename=file))


@app.route('/predict/<filename>', methods = ['GET', 'POST'])
def predict(filename):
	#return filename
	image1 = cv2.imread(filename)
	orig = image1.copy()
	# pre-process the image for classification
	#image = cv2.resize(image, (28, 28))
	image = load_img(filename,target_size=(28,28))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	#K.clear_session()
	# load the trained convolutional neural network
	print("[INFO] loading network...")
	model = load_model('model_new.h5')
	# classify the input image
	(normal, pnemonia) = model.predict(image)[0]
	# build the label
	label = "NORMAL" if normal > pnemonia else "PNEMONIA"
	proba = normal if normal > pnemonia else pnemonia
	#label = "{}: {:.2f}%".format(label, proba * 100)
	# draw the label on the image
	#K.clear_session()
	return label

if __name__ == "__main__":
    app.run(debug=True)

