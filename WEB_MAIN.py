# -*- coding: utf-8 -*-
import base64
import json
from io import StringIO, BytesIO
# from cStringIO import StringIO

import bson
import cv2
import flask
import numpy
from PIL import Image
from bson.json_util import dumps
from flask import Flask, request, Response, jsonify, redirect, json, flash, abort
from numpy import rint
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import MAIN_PROSS

# import tf_predict
# from imageFilter import ImageFilter

# import rnn_predict
# import pd_predict
import MAIN_PROSS
import shutil
import os

app = Flask(__name__, static_url_path="")
app.secret_key = '123456'

# 读取配置文件
app.config.from_object('config')

# 连接数据库，并获取数据库对象
db = MongoClient(app.config['DB_HOST'], app.config['DB_PORT']).test


# 将矫正后图片与图片识别结果（JSON）存入数据库
def save_file(file_str, f, report_data):
    #content = StringIO(file_str)
    content = file_str

    try:
        mime = 'jpeg'
        print('content of mime is：', mime)
        if mime not in app.config['ALLOWED_EXTENSIONS']:
            raise IOError()
    except IOError:
        abort(400)
    c = dict(report_data=report_data, content=file_str, filename=secure_filename(f.name),
             mime=mime)
    db.files.save(c)

    return c['_id'], c['filename']


@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            flash('No file part')
            return jsonify({"error": "No file part"})
        imgfile = request.files['imagefile']
        if imgfile.filename == '':
            flash('No selected file')
            return jsonify({"error": "No selected file"})
        if imgfile:
            # pil = StringIO(imgfile)
            # pil = Image.open(pil)
            # print 'imgfile:', imgfile
            img = cv2.imdecode(numpy.frombuffer(imgfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imwrite("tes.jpg", img)
            # todo 是否需要本地存储中转
            report_data = MAIN_PROSS.main_pross("tes.jpg", 'Feature_IMG/REPO.jpg')
            if report_data is 'ocrerr':
                data = {
                    'error':'OCR故障',
                }
                flash('OCR故障')
                return jsonify(data)
            if report_data is None:
                data = {
                    "error": '主算法意外结束',
                }
                return jsonify(data)
            #todo 开始更改
            path_img_toDB = 'temp_pics/region.jpg'

            with open(path_img_toDB, "rb") as f:
                if f is None:
                    pass
                    rint('Error! f is None!')
                else:

                    '''
                        定义file_str存储矫正后的图片文件f的内容（str格式）,方便之后对图片做二次透视以及将图片内容存储至数据库中
                    '''
                    file_str = f.read()
                    # file_str = MAIN_PROSS.cv2_to_base64(cv2.imread(path_img_toDB))
                    fid, filename = save_file(file_str, f, report_data)
            print('fid:', fid)
            if fid is not None:
                templates = "<div><img id=\'filtered-report\' src=\'/file/%s\' class=\'file-preview-image\' width=\'100%%\' height=\'512\'></div>" % (
                    fid)
                data = {
                    "templates": templates,
                }
            return jsonify(data)
            # return render_template("result.html", filename=filename, fileid=fid)
    # return render_template("error.html", errormessage="No POST methods")
    return jsonify({"error": "No POST methods"})


'''
    根据图像oid，在mongodb中查询，并返回Binary对象
'''


@app.route('/file/<fid>')
def find_file(fid):
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()
        return Response(file['content'], mimetype='image/' + file['mime'])
    except bson.errors.InvalidId:
        flask.abort(404)


'''
    直接从数据库中取出之前识别好的JSON数据，并且用bson.json_util.dumps将其从BSON转换为JSON格式的str类型
'''


@app.route('/report/<fid>')
def get_report(fid):
    # print 'get_report(fid):', fid
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()

        print('type before transform:\n', type(file['report_data']))

        report_data = bson.json_util.dumps(file['report_data'])

        print('type after transform:\n', type(report_data))
        if report_data is None:
            print('report_data is NONE! Error!!!!')
            return jsonify({"error": "can't ocr'"})
        return jsonify(report_data)
    except bson.errors.InvalidId:
        flask.abort(404)

'''
def update_report(fid, ss):
    # load json example
    with open('bloodtestdata.json') as json_file:
        data = json.load(json_file)

    for i in range(22):
        data['bloodtest'][i]['value'] = ss[i]
    json_data = json.dumps(data, ensure_ascii=False, indent=4)

    db.files.update_one({
        '_id': bson.objectid.ObjectId(fid)}, {
        '$set': {
            'report_data': json_data
        }
    }, upsert=False)

    file = db.files.find_one(bson.objectid.ObjectId(fid))
    report_data = bson.json_util.dumps(file['report_data'])
    print(report_data)
'''

def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

if __name__ == '__main__':
    shutil.rmtree('ocr_result')
    shutil.rmtree('temp_pics')
    os.mkdir('ocr_result')
    os.mkdir('temp_pics')

    app.run(host=app.config['SERVER_HOST'], port=app.config['SERVER_PORT'])

