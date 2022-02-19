# -*- coding: utf-8 -*-
import base64
import json
import time
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

import shutil
import os

import PRE_pross
import MAIN_PROSS

app = Flask(__name__, static_url_path="")
app.secret_key = '123456'

# 读取mongoDB配置文件
app.config.from_pyfile('conf\\flask_config.cfg')

# 连接数据库，并获取数据库对象
db = MongoClient(app.config['DB_HOST'], app.config['DB_PORT']).test


# 将矫正后图片与图片识别结果（JSON）存入数据库
def save_file(file_str, f, report_data):
    # content = StringIO(file_str)
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
    try:
        db.files.save(c)
    except:
        print('DB_ERR')
        abort(400)
    return c['_id'], c['filename']


@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/index.html')


@app.route('/upload', methods=['POST'])
def upload():
    time_start = time.time()
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
            img = cv2.imdecode(numpy.frombuffer(imgfile.read(), numpy.uint8), cv2.IMREAD_COLOR)
            '''
            cv2.imwrite("tes.jpg", img)
            img2 = cv2.imread('tes.jpg')
            # todo 是否需要本地存储中转
            '''

            report_data = MAIN_PROSS.main_pross(img, demo_or_not=0)
            # 判断是否报错，中文开头为错误

            err_or_not = PRE_pross.charactor_match_chinese_head(report_data)
            if err_or_not is True:
                data = {
                    'error': report_data,
                }
                flash(report_data)
                return jsonify(data)
            '''
            if report_data is 'OCR_ERR':
                data = {
                    'error': 'OCR故障',
                }
                flash('OCR故障')
                return jsonify(data)
            '''
            # todo 开始更改
            path_img_toDB = 'temp\\region.jpg'

            with open(path_img_toDB, "rb") as f:
                if f is None:
                    pass
                    rint('Error! f is None!')
                else:
                    '''
                        定义file_str存储矫正后的图片文件f的内容（str格式）,方便之后将图片内容存储至数据库中
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
            time_end = time.time()
            print('totally cost', time_end - time_start)
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

if __name__ == '__main__':
    def run():
        root_path = PRE_pross.where_is_work_folder()
        # 检查并重置工作文件夹
        try:
            shutil.rmtree('temp')
            os.mkdir('temp\\ocr_result')
            os.mkdir('temp\\DEMO')
            os.mkdir('temp\\DEMO\\mask')
        except:
            os.mkdir('temp')
            os.mkdir('temp\\DEMO')
            os.mkdir('temp\\ocr_result')
            os.mkdir('temp\\DEMO\\mask')

        app.run(host=app.config['SERVER_HOST'], port=app.config['SERVER_PORT'])


    run()
