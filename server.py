# importing libraries
from flask import Flask, request, render_template
import os
import re
import base64
import numpy as np
from PIL import Image
from werkzeug import secure_filename
import keras
from keras.models import load_model
import tensorflow as tf
#from scipy.misc import imread, imresize
from PIL import Image

# initilizing flask 
app = Flask(__name__)

# loading saved model which is train in kaggle kernel
global model
model = load_model("model/DigiModel.h5")

# initilizing graph
global graph
graph = tf.get_default_graph()

# convert canvas to image
def convertImage(imgData1):
	#print(type(imgData1))
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('test/out.png', 'wb') as output:
		output.write(base64.b64decode(imgstr))

	
# Processing request and predicting
@app.route('/find', methods =['GET', 'POST'])
def find():
	imgData = request.get_data()
	convertImage(imgData)
	img = Image.open('test/out.png')
	img = img.resize((28,28)).convert('LA')
	img = np.asarray(img)[:,:,:1]
	img = np.invert(img)
	#print(img.shape)
	img = img/ 255
	img = np.reshape(img,(1,28,28,1))
	
	with graph.as_default():
		out = model.predict(img)
	print(out)
	print(np.argmax(out, axis = 1))
	
	response = str(np.argmax(out, axis = 1))
	return response[1]

# Sending response to /predict url
@app.route('/predict')
def predict():
	return render_template('predict.html')

# Homepage
@app.route('/')
def welcome():
	return render_template('index.html')


# Starting flask server
if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug = True)

