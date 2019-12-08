from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
from keras import backend as K
import numpy as np
import random
import cv2
from keras import backend as K
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('test1.html')

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
	image = load_img(filename,target_size=(150,150))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	#K.clear_session()
	# load the trained convolutional neural network
	print("[INFO] loading network...")
	model = load_model('model_project6.h5')
	model.load_weights('weights_project6.h5')
	# classify the input image
	print("-"*100)
	result  = model.predict(image)
	results = np.array(result, dtype=float)
	pred = result[0][np.argmax(result)] * 100
	answer = np.argmax(results)
	print(pred)
	pred1 = "pnemonia"
	pred2 = "not pnemonia"
	if answer == 0:
		return render_template('test1.html',results=pred1)
	if answer == 1:
		return render_template('test1.html',results=pred2)
	#(normal, pnemonia) = model.predict(image)[0]
	# # build the label
	#label = "NORMAL" if normal > pnemonia else "PNEMONIA"
	#proba = normal if normal > pnemonia else pnemonia
	#label = "{}: {:.2f}%".format(label, proba * 100)
	# draw the label on the image
	#K.clear_session()
	#li = ['normal','pnemonia']
	#label - random.choice(li)
	#K.clear_session()
	return 0

if __name__ == "__main__":
    app.run(debug=True)

