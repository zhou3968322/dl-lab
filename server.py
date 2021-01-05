# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/5
import logging
import json
import time
import uuid
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
from utils.log import logger, set_log_file
from flask import request, Flask, send_from_directory, make_response
from flask_cors import CORS
import anyconfig
import predictor
from PIL import Image
from utils.common_util import get_predictor_name

app = Flask(__name__)

CORS(app, supports_credentials=True)
app.secret_key = 'YhUTAie0d8'
app.config['JSON_AS_ASCII'] = False

PROJECT_DIR = os.path.abspath(os.path.join(__file__, "../"))
DATA_DIR = os.path.join(PROJECT_DIR, "online_data")
UPLOAD_DIR = os.path.join(DATA_DIR, "upload")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config['upload'] = UPLOAD_DIR
app.config['output'] = OUTPUT_DIR

config_file_path = os.path.join(PROJECT_DIR, "config/gan/medfe_server.yaml")

config = anyconfig.load(open(config_file_path, 'rb'))
server_predictor = getattr(predictor, get_predictor_name("medfe"))(config)

log_path = os.path.join(PROJECT_DIR, "server.log")
set_log_file(log_path)


@app.route('/watermark', methods=['POST'])
def remove_watermark():
    if request.method == 'POST':
        try:
            bt = time.time()
            logger.info("start process handle water mark,pid:{}".format(os.getpid()))
            files = request.files.getlist('file')
            if not files:
                logger.info('No file upload')
                return 'No file upload', 400
            if len(files) > 1:
                return 'not support batch file upload', 400
            file_stream = files[0]
            if not is_support_file(file_stream):
                return 'not support file ext', 400
            old_name = file_stream.filename
            logger.info('start process old name {},pid:{}'.format(old_name, os.getpid()))
            input_file = save_upload_file(files[0])
            logger.info('current process file name {}'.format(input_file))
            base_name = os.path.basename(input_file)
            output_mask_name = "{}_mask.jpeg".format(base_name.rsplit('.', 1)[0])
            output_mask_path = os.path.join(app.config["output"], output_mask_name)
            output_img_path = os.path.join(app.config["output"], base_name)
            img = Image.open(input_file)
            mask_im, fake_im = server_predictor.predict_single(single_img=img)
            mask_im.save(output_mask_path)
            fake_im.save(output_img_path)
            res_data = {
                "mask_img_name": output_mask_name,
                "output_img_name": base_name
            }
            logger.info('success process file {},cost {}, pid:{}'.format(old_name, time.time() - bt, os.getpid()))
            return json.dumps(res_data, ensure_ascii=False), 200
        except Exception as e:
            logger.exception('error handling, status:500')
            return str(e), 500


@app.route('/file/<filename>', methods=['GET'])
def get_file(filename):
    return send_from_directory(app.config['output'], filename)


def is_support_img(file_stream):
    full_file_name = file_stream.filename
    file_name, file_ext = full_file_name.rsplit('.', 1)
    return file_ext.lower() in ['jpg', 'png', 'tif', 'jpeg']


def is_pdf(file_stream):
    full_file_name = file_stream.filename
    file_name, file_ext = full_file_name.rsplit('.', 1)
    return file_ext.lower() in ['pdf']


def is_support_file(file_stream):
    full_file_name = file_stream.filename
    file_name, file_ext = full_file_name.rsplit('.', 1)
    return file_ext.lower().strip("\"") in ['jpg', 'png', 'tif', 'jpeg']


def get_output_file_stream(output_file):
    base_directory = os.path.dirname(output_file)
    filename = os.path.basename(output_file)
    response = make_response(
        send_from_directory(base_directory, filename, as_attachment=True))
    response.headers[
        "Content-Disposition"] = "attachment; filename={}".format(filename)
    return response


def save_upload_file(file_stream):
    full_file_name = file_stream.filename
    ori_file_name, file_ext = full_file_name.rsplit('.', 1)
    unique_name = str(uuid.uuid1().hex)
    unique_file = unique_name + '.' + file_ext.lower()
    upload_save_path = os.path.join(app.config['upload'], unique_file)
    file_stream.save(upload_save_path)
    file_stream.close()
    return upload_save_path


def run():
    app.run(port=8400, host="0.0.0.0", threaded=False, debug=False)


if __name__ == '__main__':
    run()