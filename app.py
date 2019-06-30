from flask import Flask, request,render_template,url_for,session,send_file,Response,send_from_directory
from flask_restful import  Api
import json
import uuid
import numpy as np
import logging
from flask.logging import default_handler
import os
from io import BytesIO
from PIL import Image,ImageDraw
#import cv2 as cv2
from tensorflow.keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.models import model_from_json


app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'uploads'
PREDICTIONS_FOLDER = 'predictions'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
   
#Loading the model    
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
set_session(sess)
json_file = open('model/model2_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#load architecture
model = model_from_json(loaded_model_json)
# load weights into model
model.load_weights("model/model2.h5")


if __name__ == 'app':
    #File handler. Remove default handler    
    
    logHandler = logging.FileHandler('application_logs.log')
    logHandler.setLevel(logging.INFO)
    logHandler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    app.logger.removeHandler(default_handler)        
    app.logger.addHandler(logHandler)
    app.logger.setLevel(logging.INFO)    
    app.run(debug=False)


@app.route("/")
def default():
    return render_template('home.html')


@app.route("/upload",methods=["GET","POST"])
def upload():    
    file = request.files['file']
    file_name=file.filename
    uid=uuid.uuid4().hex+(file_name[file_name.rfind('.'):])
    file.save(os.path.join(app.config['UPLOAD_FOLDER'],uid))
    response={'status':'200','uid':uid}
    return json.dumps(response)
    

@app.route("/result",methods=["GET","POST"])
def showResult():
    
    key=request.form['key']
    uploadedimage=Image.open(os.path.join(app.config['UPLOAD_FOLDER'],key))
    uploadedimage=uploadedimage.resize(size=(256,256))
    scaledimage=np.array(uploadedimage)/255.
    pred=predict(scaledimage)
    if(np.count_nonzero(pred.flatten())>100):#atleast 100 pixels should be classified as fake
        
        indices = np.where(pred[0,:,:,0] == 1)
        upper = np.min(indices[0])
        lower = np.max(indices[0])
        left = np.min(indices[1])
        right = np.max(indices[1])
        '''
        temp=np.array(uploadedimage)
        r,g,b=cv2.split(temp)
        bgrimage=cv2.merge([b,g,r])
        result=cv2.rectangle(bgrimage,(left,upper),(right,lower),(0,255,0),2)
        prefix=key[0:key.rfind('.')]
        suffix=key[key.rfind('.'):]
        result_name=prefix+'.predict'+suffix
        cv2.imwrite(os.path.join(app.config['PREDICTIONS_FOLDER'],result_name),result)
        '''
        draw = ImageDraw.Draw(uploadedimage)
        draw.rectangle([(left,upper),(right,lower)],outline=(0,255,0),width=3)
        msg='Image Classified as Fake'
    else:
        msg='Image Classified as Pristine'
        
    prefix=key[0:key.rfind('.')]
    suffix=key[key.rfind('.'):]
    result_name=prefix+'.predict'+suffix
    uploadedimage.save(os.path.join(app.config['PREDICTIONS_FOLDER'],result_name),quality=100)
   
    response={'status':'200','uid':key,'result':result_name,'msg':msg}
    return json.dumps(response)


@app.route('/predictions/<path:filename>')  
def send_file(filename):  
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)


def predict(img):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        global model
        pred=model.predict(img.reshape(1,256,256,3))
    pred= (pred > 0.5).astype(np.uint8)    
    return pred
