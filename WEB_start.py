from flask import Flask,request
from scipy import misc
import cv2
import numpy as np
import numpy
from flask import Flask, request, Response, jsonify, redirect, json, flash

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    imgfile = request.files['imagefile']
    if imgfile.filename == '':
        flash('No selected file')
        return jsonify({"error": "No selected file"})
    if imgfile:
        # pil = StringIO(imgfile)
        # pil = Image.open(pil)
        # print 'imgfile:', imgfile
        img = cv2.imdecode(numpy.fromstring(imgfile.read(), numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


    return 'predict: ok '

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''
if __name__ == '__main__':
    app.run()
