from pprint import pprint
# from paddleocr import PaddleOCR, draw_ocr
import requests
import json
import cv2
import base64
# from PIL import Image
import configparser
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import PRE_pross
import shutil
import os

from collections import defaultdict, OrderedDict


class configparser_custom(configparser.ConfigParser):  # 解决默认被转换为小写问题
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


def net_OCR(cvimg):
    # 发送HTTP请求
    data = {'images': [cv2_to_base64(cvimg)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/chinese_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    # pprint(r)
    return r


# def main_pross(img_orig_path, img_feature_path):
def main_pross(img_orig_path, img_feature_path):
    # img_orig_path = 'Input_IMG/Input.jpg'
    # img_feature_path = 'Feature_IMG/REPO.jpg'
    # img_path = 'REPO1.jpg'
    # img_feature = cv2.imread(img_feature_path)

    img_org = cv2.imread(img_orig_path)

    # img_org = img

    img_gamma = PRE_pross.gamma(img_org)

    # 读取配置
    conf_path = 'conf/res1k.conf'
    conf = configparser_custom()
    conf.read(conf_path, 'UTF-8')
    boxes_conf = conf.items("boxes")
    name_list = []
    box_list = []
    for w in range(len(boxes_conf)):
        name_list.append(boxes_conf[w][0])
        box_list.append(boxes_conf[w][1].split(','))
        box_list[w] = list(map(int, box_list[w]))

    # 特征匹配准备裁剪
    img_template = cv2.imread(img_feature_path, 0)
    img_need_pross = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

    img_small_1k, ratio = PRE_pross.zoom_to_1k(img_need_pross)  # 屏幕匹配提速

    # [旧]correct_points, knn_result = knn_match_old(img_template, img_small_1k, demo)
    correct_matrix, knn_result = PRE_pross.knn_match_new(img_template, img_small_1k, 0)
    if knn_result == 2874734:
        return '不甚匹配'
    # 以下是新变换办法，直接用单应性矩阵变换后直接裁剪得到目标图像
    correct_matrix[0][2] = correct_matrix[0][2] / ratio
    correct_matrix[1][2] = correct_matrix[1][2] / ratio
    correct_matrix[2][0] = correct_matrix[2][0] * ratio
    correct_matrix[2][1] = correct_matrix[2][1] * ratio
    img_screen_cut = cv2.warpPerspective(img_need_pross, correct_matrix, (round((img_template.shape[1]) / ratio),
                                                                          round((img_template.shape[0]) / ratio)))
    # cv2.imwrite('res.jpg', img_screen_cut)
    # plt.imshow(img_screen_cut, 'gray'), plt.show()
    img_screen_cut_1k, ratio_outdate = PRE_pross.zoom_to_1k(img_screen_cut)

    cv2.imwrite('temp_pics/region.jpg', img_screen_cut_1k)  # 适配开源血常规

    # img_screen_cut, ratio_2k = zoom_to_2k(img_screen_cut)  # 放大以方便识别

    # cv2.imwrite('TEMP/2k_result.jpg', img_screen_cut)
    img_screen_cut = PRE_pross.length_width_ratio_correct(img_template, img_screen_cut)  # 长宽比校正

    # 下面开始按照比例裁剪识别区域
    img_element = PRE_pross.mask_processing_new(img_screen_cut,
                                                box_list,
                                                demo_or_not=0,
                                                type_char='repo',
                                                out_name='report',
                                                output_dir='DEMO/')  # 根据配置裁剪数据区
    # 拓宽防止ocr不识别的bug
    for i in range(len(img_element)):
        img_element[i] = PRE_pross.image_border(img_element[i], '0')
    # 取得识别结果
    answer = []
    for n in range(len(img_element)):
        response = net_OCR(img_element[n])
        if len(response.json()["results"])==0:
            print('ocrerr')
            return 'ocrerr'
        print(response.json()["results"][0]["data"])
        answer.append(response)
        answer[n] = answer[n].json()["results"]

    # 判断是否识别完全
    count = []
    for n in range(len(img_element)):
        count.append(len(answer[n][0]['data']))

    if count[0] == count[1] and count[1] == count[2]:
        if count[3] == count[4] and count[4] == count[5]:
            print('无缺损')
        else:
            print('右边识别不完全')
    else:
        print('左边识别不完全')

    print(answer[1][0]['data'][1]['text'])
    # 以下开始处理识别回传数据
    name_out = []
    value_out = []
    range_out = []
    for i in range(len(answer)):
        if i == 0 or i == 3:
            for j in range(len(answer[i][0]['data'])):
                name_out.append(answer[i][0]['data'][j]['text'])
        if i == 1 or i == 4:
            for j in range(len(answer[i][0]['data'])):
                value_out.append(answer[i][0]['data'][j]['text'])
        if i == 2 or i == 5:
            for j in range(len(answer[i][0]['data'])):
                range_out.append(answer[i][0]['data'][j]['text'])
    else:
        print('err')
    # 以下开始json化
    bloodtest_list = []
    for i in range(len(name_out)):
        bloodtest_single = OrderedDict()
        bloodtest_single["name"] = name_out[i]
        bloodtest_single["value"] = value_out[i]
        bloodtest_single["range"] = range_out[i]
        bloodtest_single["alias"] = "Empty"
        bloodtest_single["unit"] = "Empty"
        bloodtest_list.append(bloodtest_single)
    test_dict = {
        'version': "1.0",
        'bloodtest': bloodtest_list,
        'explain': {
            'used': True,
            'details': "this is for josn test",
        }
    }
    json_str = json.dumps(test_dict, ensure_ascii=False, indent=4)
    with open('test_data.json', 'w') as json_file:
        json_file.write(json_str)
    return json_str



if __name__ == '__main__':
    # 清空OCR
    shutil.rmtree('ocr_result')
    os.mkdir('ocr_result')
    img_orig_path = 'Input_IMG/Input.jpg'
    img_feature_path = 'Feature_IMG/REPO.jpg'
    main_pross(img_orig_path, img_feature_path)
