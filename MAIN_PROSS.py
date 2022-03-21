# coding=utf-8

# from PIL import Image
from pprint import pprint
from glob import glob
import re
import requests
import json
import cv2
import base64
import matplotlib.pyplot as plt

from colorama import init, Fore, Back
init(autoreset=True)

plt.switch_backend('agg')
import shutil
import os
from collections import defaultdict, OrderedDict
import PRE_pross
import configparser


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
    try:
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
    except ConnectionRefusedError:
        r = 'OCROFFLINE'
        print(r)
    except requests.exceptions.ConnectionError:
        r = 'OCROFFLINE'
        print(r)
    # pprint(r)
    return r


def main_pross(cvimg, demo_or_not):
    img_org = cvimg
    img_gamma = PRE_pross.gamma(img_org)

    # 判断所属医院以及检验项目
    img_gamma = PRE_pross.image_border(img_input=img_gamma,
                                       dst='0')
    pre_response = net_OCR(img_gamma)
    if pre_response is 'OCROFFLINE':
        return '错误：OCR离线'
    elif len(pre_response.json()["results"]) == 0:
        # print('OCRERR')
        return f'错误：OCR没有正常工作,{pre_response.text}'
    # pprint(pre_response.json()["results"][0]["data"])
    report_overview = []
    for i in range(len(pre_response.json()["results"][0]["data"])):
        report_overview.append(pre_response.json()["results"][0]["data"][i]['text'])

    # 取出医院关键词
    hospital = PRE_pross.charactor_match_hospital_name(report_overview, '医院')
    path_prefix = hospital
    # ocr会把间隔大的文字分开识别，大概率优先识别为中文字符
    '''
    patient_name = PRE_pross.charactor_match_count_name_age(report_overview, '名：')
    if patient_name:
        # 以防万一把所有冒号前的东西重新统一
        patient_name = re.sub(r'.*：', '姓名：', patient_name)
    else:
        patient_name = '姓名：'
    patient_sex = PRE_pross.charactor_match_count_sex(report_overview, '别：')
    if patient_sex:
        patient_sex = re.sub(r'.*：', '性别：', patient_sex)
    else:
        patient_name = '性别：'
    patient_age = PRE_pross.charactor_match_count_name_age(report_overview, '龄：')
    if patient_age:
        patient_age = re.sub(r'.*：', '年龄：', patient_age)
    else:
        patient_name = '年龄：'
    '''
    if hospital is None:
        # print('未能识别医院信息')
        return '错误：未能识别所属医院，请拍摄完整的报告单图片，并保证纸面平整'

    # 读取报告类型关键词
    '''
    def read_keywords(type):
        keywords_conf_path = f'conf/[关键词]{type}.conf'
        keys = configparser_custom()
        keys.read(keywords_conf_path, 'UTF-8')
        keys_read = keys.items("keywords")
        # keys_list = []
        return keys_read
    '''
    def read_keywords(path):
        #keywords_conf_path = f'conf/[关键词]{type}.conf'
        keys = configparser_custom()
        keys.read(path, 'UTF-8')
        keys_read = keys.items("keywords")
        # keys_list = []
        return keys_read

    '''
    def type_judge(keys_need_judge, keys_read, text):
        n = 0.00
        keys_list = []
        for w in range(len(keys_read)):
            keys_list.append(keys_read[w][1])
            if PRE_pross.charactor_match_count_name_age(keys_need_judge, keys_list[w]):
                n = n + 1.00
        if n >= (float(len(keys_read))) / 2:
            print(f"-是{text}-")
            path_suffix = f'-{text}'
            return path_suffix
        else:
            print(f'-不是{text}-')
            #return '错误：目前不支持此种报告，请勿上传其他类型报告'
            return False
    '''
    def type_judge(lstKwds_need_judge, conf_path):
        path = f"{conf_path}/[*.conf"
        lstTxtFiles = glob(path)
        for strTxtFile in lstTxtFiles:
            keys_list = read_keywords(strTxtFile)
            keys_list_t = []
            for w in range(len(keys_list)):
                keys_list_t.append(keys_list[w][1])
            # strContent = txtWrapper.read()  # 读关键词
            i = 0.00
            n = 0.00
            for strKwd in keys_list_t:  # 用每个从本地读取到的关键词去匹配
                n = len(keys_list_t)
                if PRE_pross.charactor_match_any(lstKwds_need_judge, strKwd):
                    # if strKwd in strContent:  # 如果命中
                    i = i + 1
            print(i / n)
            if (i / n) > 0.4:
                # print(os.path.basename(strTxtFile))
                find = os.path.basename(strTxtFile)
                find_no_ex = find.split('.conf')
                find_type = find_no_ex[0].split(']')
                type = find_type[1]
                print(Back.GREEN+type)
                return type

    report_type = type_judge(lstKwds_need_judge=report_overview,
                             conf_path='conf')
    path_suffix = f'-{report_type}'

    '''
    n = 0.00
    for w in range(len(blood_keys_read)):
        blood_keys_list.append(blood_keys_read[w][1])
        if PRE_pross.charactor_match_count_name_age(report_overview, blood_keys_list[w]):
            n = n + 1.00
    if n >= (float(len(blood_keys_read))) / 2:
        print("-是血常规-")
        is_blood_test = "是血常规"
        path_suffix = '-血常规'
    else:
        print('-不是血常规-')
        is_blood_test = "不是血常规"
        return '错误：目前仅支持血常规，请勿上传其他类型报告'
    '''
    conf_path = f'conf/{path_prefix}{path_suffix}.conf'
    if os.path.exists(conf_path) is False:
        # print('配置文件不存在')
        return '错误：目前暂不支持此医院的此种报告'
    img_feature_path = f'OCR_IMG/Feature_IMG/{path_prefix}{path_suffix}.jpg'
    if os.path.exists(img_feature_path) is False:
        # print('特征图片不存在')
        return '错误：缺少特征图片，无法匹配，请等候开发者后续维护'

    # 读取配置
    # conf_path = 'conf/bj-aerospace-blood-normal.conf'
    conf = configparser_custom()
    conf.read(conf_path, 'UTF-8')
    boxes_conf = conf.items("boxes")
    name_list = []
    # box_list[] 格式:[左，右，上，下]
    box_list = []
    for w in range(len(boxes_conf)):
        name_list.append(boxes_conf[w][0])
        box_list.append(boxes_conf[w][1].split(','))
        box_list[w] = list(map(int, box_list[w]))

    # 特征匹配准备裁剪
    # img_template = cv2.imread(img_feature_path, 0)
    img_template = PRE_pross.cv_imread_chs(img_feature_path)
    # 灰度化
    img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    img_need_pross = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

    img_small_1k, ratio = PRE_pross.zoom_to_1k(img_need_pross)  # 屏幕匹配提速

    # [旧]correct_points, knn_result = knn_match_old(img_template, img_small_1k, demo)
    correct_matrix, knn_result = PRE_pross.knn_match_new(template_img=img_template,
                                                         img_need_match=img_small_1k,
                                                         demo=demo_or_not)
    if knn_result == 2874734:
        return '错误：未能成功探测布局，建议重新拍摄'
    # 以下是新变换办法，直接用单应性矩阵变换后直接裁剪得到目标图像（能解决老办法不能自动旋转的问题）
    correct_matrix[0][2] = correct_matrix[0][2] / ratio
    correct_matrix[1][2] = correct_matrix[1][2] / ratio
    correct_matrix[2][0] = correct_matrix[2][0] * ratio
    correct_matrix[2][1] = correct_matrix[2][1] * ratio
    img_screen_cut = cv2.warpPerspective(img_need_pross, correct_matrix, (round((img_template.shape[1]) / ratio),
                                                                          round((img_template.shape[0]) / ratio)))
    # cv2.imwrite('res.jpg', img_screen_cut)
    # plt.imshow(img_screen_cut, 'gray'), plt.show()
    img_screen_cut_1k, ratio_outdate = PRE_pross.zoom_to_1k(img_screen_cut)

    cv2.imwrite('temp/region.jpg', img_screen_cut_1k)  # 适配开源血常规

    # 将用户信息识别滞后，提升识别概率
    usr_info_response = net_OCR(img_screen_cut_1k)
    if usr_info_response is 'OCROFFLINE':
        return '错误：OCR离线'
    elif len(usr_info_response.json()["results"]) == 0:
        # print('OCRERR')
        return f'错误：OCR没有正常工作,\n{usr_info_response.text}'
    # pprint(usr_info_response.json()["results"][0]["data"])
    usr_info_overview = []
    for i in range(len(usr_info_response.json()["results"][0]["data"])):
        usr_info_overview.append(usr_info_response.json()["results"][0]["data"][i]['text'])
    patient_name = PRE_pross.charactor_match_count_name_age(usr_info_overview, '名：')
    if patient_name:
        # 以防万一把所有冒号前的东西重新统一
        patient_name = re.sub(r'.*：', '姓名：', patient_name)
    else:
        patient_name = '姓名：'
    patient_sex = PRE_pross.charactor_match_count_sex(usr_info_overview, '别：')
    if patient_sex:
        patient_sex = re.sub(r'.*：', '性别：', patient_sex)
    else:
        patient_name = '性别：'
    patient_age = PRE_pross.charactor_match_count_name_age(usr_info_overview, '龄：')
    if patient_age:
        patient_age = re.sub(r'.*：', '年龄：', patient_age)
    else:
        patient_name = '年龄：'
    # 用户信息识别结束

    img_screen_cut = PRE_pross.length_width_ratio_correct(img_template=img_template,
                                                          img_input=img_screen_cut)  # 长宽比校正

    # 下面开始按照比例裁剪识别区域
    img_element = PRE_pross.mask_processing_new(img_input=img_screen_cut,
                                                boxes_coordinate_xy=box_list,
                                                demo_or_not=demo_or_not,
                                                type_char='repo',
                                                out_name='report')  # 根据配置裁剪数据区
    # 拓宽防止ocr不识别的bug
    for i in range(len(img_element)):
        img_element[i] = PRE_pross.image_border(img_input=img_element[i],
                                                dst='0')
    # 取得识别结果
    answer = []
    for n in range(len(img_element)):
        response = net_OCR(img_element[n])
        if response is 'OCROFFLINE':
            return '错误：OCR离线'
        elif len(response.json()["results"]) == 0:
            # print('OCRERR')
            return f'错误：OCR未正常工作,{response.text}'

        # pprint(response.json()["results"][0]["data"])
        answer.append(response)
        answer[n] = answer[n].json()["results"]

    # 判断是否识别完全
    # 以下为老代码，暂时废弃
    '''
    count = []
    for n in range(len(img_element)):
        count.append(len(answer[n][0]['data']))

    if count[0] == count[1] and count[1] == count[2]:
        if count[3] == count[4] and count[4] == count[5]:
            print(Fore.LIGHTBLUE_EX+'无缺损')
        else:
            print(Back.RED+'右边识别不完全')
            return '错误：识别异常，数据量不匹配，建议重新拍摄'
    else:
        print(Back.RED+'左边识别不完全')
        return '错误：识别异常，数据量不匹配，建议重新拍摄'
    '''
    # print(answer[1][0]['data'][1]['text'])
    # 以下开始处理识别回传数据
    # 格式说明： ?_position[],先长后高，左上开始顺时针4点
    name_out = []
    name_out_position = []
    value_out = []
    value_out_position = []
    range_out = []
    range_out_position = []
    name_out2 = []
    name_out_position2 = []
    value_out2 = []
    value_out_position2 = []
    range_out2 = []
    range_out_position2 = []
    for i in range(len(answer)):
        if i == 0:
            for j in range(len(answer[i][0]['data'])):
                name_out.append(answer[i][0]['data'][j]['text'])
                name_out_position.append(answer[i][0]['data'][j]['text_box_position'])
        if i == 1:
            for j in range(len(answer[i][0]['data'])):
                value_out.append(answer[i][0]['data'][j]['text'])
                value_out_position.append(answer[i][0]['data'][j]['text_box_position'])
        if i == 2:
            for j in range(len(answer[i][0]['data'])):
                range_out.append(answer[i][0]['data'][j]['text'])
                range_out_position.append(answer[i][0]['data'][j]['text_box_position'])
        if i == 3:
            for j in range(len(answer[i][0]['data'])):
                name_out2.append(answer[i][0]['data'][j]['text'])
                name_out_position2.append(answer[i][0]['data'][j]['text_box_position'])
        if i == 4:
            for j in range(len(answer[i][0]['data'])):
                value_out2.append(answer[i][0]['data'][j]['text'])
                value_out_position2.append(answer[i][0]['data'][j]['text_box_position'])
        if i == 5:
            for j in range(len(answer[i][0]['data'])):
                range_out2.append(answer[i][0]['data'][j]['text'])
                range_out_position2.append(answer[i][0]['data'][j]['text_box_position'])
    else:
        pass
        # print('err')

    if len(name_out) == 0:
        print(Back.RED+'无数据')
        return '错误：没有识别到有效数据'

    # 开始根据坐标对齐
    def data_align(list_name, list_name_position, list_value, list_value_position, list_range, list_range_position):
        list_out = []
        for i in range(len(list_name)):
            list_out_child = []
            height_up = 0.5*(float(list_name_position[i][0][1])+float(list_name_position[i][1][1]))
            height_down = 0.5 * (float(list_name_position[i][2][1]) + float(list_name_position[i][3][1]))
            height_should = 0.5*(height_up+height_down)

            list_out_child.append(list_name[i])

            for j in range(len(list_value)):
                height_up2 = 0.5 * (float(list_value_position[j][0][1]) + float(list_value_position[j][1][1]))
                height_down2 = 0.5 * (float(list_value_position[j][2][1]) + float(list_value_position[j][3][1]))
                height_should2 = 0.5 * (height_up2 + height_down2)

                if abs(height_should-height_should2) <= 5.0:
                    list_out_child.append(list_value[j])
                    break
                else:
                    pass
            if len(list_out_child) == 1:
                list_out_child.append('空')

            for k in range(len(list_range)):
                height_up3 = 0.5 * (float(list_range_position[k][0][1]) + float(list_range_position[k][1][1]))
                height_down3 = 0.5 * (float(list_range_position[k][2][1]) + float(list_range_position[k][3][1]))
                height_should3 = 0.5 * (height_up3 + height_down3)
                if abs(height_should-height_should3) <= 5.0:
                    list_out_child.append(list_range[k])
                    break
                else:
                    pass
            if len(list_out_child) == 2:
                list_out_child.append('空')
            list_out.append(list_out_child)
        return list_out

    list_new = data_align(name_out, name_out_position, value_out, value_out_position, range_out, range_out_position)
    list_new2 = data_align(name_out2, name_out_position2, value_out2, value_out_position2, range_out2, range_out_position2)
    list_new.extend(list_new2)

    # 旧 数据复原等待处理
    '''
    name_out.extend(name_out2)
    name_out_position.extend(name_out_position2)
    value_out.extend(value_out2)
    value_out_position.extend(value_out_position2)
    range_out.extend(range_out2)
    range_out_position.extend(range_out_position2)
    '''

    # 旧 以下开始json化
    '''
    bloodtest_list = []
    for i in range(len(name_out)):
        bloodtest_single = OrderedDict()
        bloodtest_single["name"] = name_out[i]
        bloodtest_single["value"] = value_out[i]
        bloodtest_single["range"] = range_out[i]
        bloodtest_single["alias"] = ''
        bloodtest_single["unit"] = ''
        bloodtest_list.append(bloodtest_single)
    '''
    # 新 JSON化
    bloodtest_list = []
    for i in range(len(list_new)):
        bloodtest_single = OrderedDict()
        bloodtest_single["name"] = list_new[i][0]
        bloodtest_single["value"] = list_new[i][1]
        bloodtest_single["range"] = list_new[i][2]
        bloodtest_single["alias"] = ''
        bloodtest_single["unit"] = ''
        bloodtest_list.append(bloodtest_single)

    # 加入附加信息
    bloodtest_single = OrderedDict()
    bloodtest_single["name"] = f'{hospital}{path_suffix}'
    bloodtest_single["value"] = patient_name
    bloodtest_single["range"] = patient_sex
    bloodtest_single["alias"] = '空白信息2'
    bloodtest_single["unit"] = patient_age
    bloodtest_list.append(bloodtest_single)
    test_dict = {
        'version': "0.3",
        'bloodtest': bloodtest_list,
        'explain': {
            'used': True,
            'details': "json生成测试",
        }
    }
    json_str = json.dumps(test_dict, ensure_ascii=False, indent=4)
    with open('test_data.json', 'w') as json_file:
        json_file.write(json_str)
    return json_str


if __name__ == '__main__':
    # 清空临时文件
    shutil.rmtree('temp')
    os.mkdir('temp')
    os.mkdir('temp/DEMO')
    os.mkdir('temp/ocr_result')
    os.mkdir('temp/DEMO/mask')

    img_orig_path = 'OCR_IMG/Input_IMG/zs-blood-normal.jpg'
    img_input = PRE_pross.cv_imread_chs(img_orig_path)

    main_pross(img_input, demo_or_not=1)
