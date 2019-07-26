from flask import Flask, request,render_template,url_for,session,send_file,Response,send_from_directory
from flask_restful import  Api
import json
import numpy as np
from PIL import Image,ImageDraw
from tensorflow.keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import ExifTags
import base64
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import traceback


app = Flask(__name__)
api = Api(app)

   
#Loading the model    
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
set_session(sess)
json_file = open('model/model3_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#load architecture
model = model_from_json(loaded_model_json)
# load weights into model
model.load_weights("model/model3.h5")


@app.route("/")
def default():
    return render_template('home.html')

@app.route("/upload",methods=["POST"])
def upload():
    try:
        print('in upload')
        file = request.files['file'].read()
        uploadedimage=Image.open(BytesIO(file))
        
        verifyExif=checkExifData(uploadedimage)
        uploadedimage=uploadedimage.resize(size=(256,256))
        if verifyExif==False:
            scaledimage=np.array(uploadedimage)/255.
            pred=predict(scaledimage)
            if(np.count_nonzero(pred.flatten())>100):#atleast 100 pixels should be classified as fake
                
                indices = np.where(pred[0,:,:,0] == 1)
                upper = np.min(indices[0])
                lower = np.max(indices[0])
                left = np.min(indices[1])
                right = np.max(indices[1])
                
                draw = ImageDraw.Draw(uploadedimage)
                draw.rectangle([(left,upper),(right,lower)],outline=(0,255,0),width=3)
                msg='Image Classified as Fake'
                googleurl=reverseImageSearch(file)
                content=scrapeGoogleResults(googleurl)
                if len(content)>0:
                    print('Scraping Data Exist')
            else:
                msg='Image Classified as Pristine'
                content=[]
        else:
            msg='Image Classified as Pristine'
            content=[]
        byte_io = BytesIO()
        uploadedimage.save(byte_io,'JPEG',quality=100)
        byte_io.seek(0)
        response={'status':'200','msg':msg,'prediction':base64.b64encode(byte_io.getvalue()).decode(),'content':content}
    except:
        print(traceback.format_exc())
        response={'status':'500','msg':'Some error occurred'}
    return json.dumps(response)
    

def predict(img):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        global model
        pred=model.predict(img.reshape(1,256,256,3))
    pred= (pred > 0.5).astype(np.uint8)    
    return pred


def checkExifData(img):
    toReturn=False
    if img.format=='JPEG':
        exifDataRaw = img._getexif()
        if exifDataRaw!=None:
            exifData = {}
            for tag, value in exifDataRaw.items():
                decodedTag = ExifTags.TAGS.get(tag, tag)
                exifData[decodedTag] = value
            if exifData.get('Software')==None or checkForForgerySoftware(exifData.get('Software'))==False:
                toReturn=True
    return toReturn

def checkForForgerySoftware(softwarename):
    softwares=['PHOTOSHOP','GIMP','PAINT']
    return any(softwarename.upper().find(sw)!=-1 for sw in softwares)
	
	
def reverseImageSearch(img):
    
    fetchUrl=None
    try:
        searchUrl = 'http://www.google.hr/searchbyimage/upload'
        multipart = {'encoded_image':img, 'image_content': ''}
        response = requests.post(searchUrl, files=multipart, allow_redirects=False)
        fetchUrl = response.headers['Location']
    except:
        print('Error during reverse image search')
        print(traceback.format_exc())
    return fetchUrl

def scrapeGoogleResults(fetchUrl,maxResults=3):
    print('scrapeGoogleResults')
    print(fetchUrl)
    toReturn=[]
    if fetchUrl==None or fetchUrl=='':
        return toReturn
    try:
        
        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        res=requests.get(fetchUrl,headers=headers)
        print('result')
        print(res.text)
        soup=BeautifulSoup(res.text,'html.parser')
        if soup!=None:        
            div=soup.find_all('div',attrs={'class':'srg'})        
            if div!=None and len(div)>0:
                divtofetch=0 if len(div)==1 else 1
                results=div[divtofetch].find_all('div',attrs={'class':'g'})
                count=1
                for result in results:
                    if count>maxResults:
                        break
                    obj={}
                    text=result.find_all('div',attrs={'class':'r'})
                    imgdiv=result.find_all('div',attrs={'class':'s'})
                    if(len(text)>0):
                        shortDesc=text[0].find_all('h3')
                        link=text[0].find_all('a')
                        obj['shortDesc']=shortDesc[0].text
                        obj['link']=link[0].get('href')                    
                    if(len(imgdiv)>0):    
                        smallimg=imgdiv[0].find_all('img')
                        if len(smallimg)==0:
                            continue
                        else:
                            obj['smallimg']=smallimg[0].get('src')
                    toReturn.append(obj)
                    count+=1
    except:
        print('Error while scraping url:',fetchUrl)
        print(traceback.format_exc())
    return toReturn
            
 