# coding=utf-8

# from PIL import Image
from pprint import pprint
from glob import glob
from string import digits
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

    def as_dict(self):
        """
        将configparser.ConfigParser().read()读到的数据转换成dict返回
        :return:
        """
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


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


def data_align_new(input_body_list, list_title):
    '''
    此处导入识别主体list和每一列的数据名称
    :return: list
    '''
    # todo 如何改进定义严格程度使更精准（单位像素）
    judge = 13
    # 初始化输出列表
    list_out = []
    dict_out = []

    # 确定每行共有多少列
    number_of_columns = len(input_body_list)  # 此时input_list[numb...]和input_list[0]为检测项目名字，即对齐依据

    # number_of_columns = int(len(input_list)/2)  # 此时input_list[numb...]和input_list[0]为检测项目名字，即对齐依据

    def get_h_location_func(child_list_input):
        list_name_h_location_output = []
        for i in range(len(child_list_input)):  # 遍历第一次识别的所有结果

            h_u = 0.5 * (float(child_list_input[i]['text_box_position'][0][1]) +
                         float(child_list_input[i]['text_box_position'][1][1]))
            h_d = 0.5 * (float(child_list_input[i]['text_box_position'][2][1]) +
                         float(child_list_input[i]['text_box_position'][3][1]))
            h_location = 0.5 * (h_u + h_d)  # 确定每一项的高度位置
            list_name_h_location_output.append(h_location)
        return list_name_h_location_output

    list_name_h_location = get_h_location_func(input_body_list[0][0]['data'])

    # 解决一行名字被识别为两项的情况（名字中间有过长空格）

    list_invalid_name = []  # 记录无效项
    for i in range(len(list_name_h_location)):
        if i in list_invalid_name:
            pass
        else:
            for j in range(len(list_name_h_location)):
                if j <= i:
                    pass
                else:
                    if abs(list_name_h_location[i] - list_name_h_location[j]) < judge:
                        list_invalid_name.append(j)

    list_name_correct = []
    list_name_position_correct = []
    for i in range(len(input_body_list[0][0]['data'])):
        if i in list_invalid_name:
            for n in range(4):  # 往前推，合并到最近的正常项
                if i - 1 - n in list_invalid_name:
                    pass
                else:
                    list_name_correct[-1] = list_name_correct[-1] + ' ' + (
                        input_body_list[0][0]['data'][i]['text'])  # 向前合并
                    break
            pass
        else:
            list_name_correct.append(input_body_list[0][0]['data'][i]['text'])  # 第一次组合，先填入行名字(检测项目名)
            list_name_position_correct.append(list_name_h_location[i])
    for i in range(len(list_name_correct)):  # 以下开始组合每一条数据
        list_out_child = []

        dict_out_child = dict()

        list_out_child.append(list_name_correct[i])  # 确定第一项：名字

        dict_out_child[list_title[0]] = list_name_correct[i]

        for k in range(1, number_of_columns):

            if len(input_body_list[k][0]['data']) == 0:
                list_out_child.append('空')

                dict_out_child[list_title[k]] = '空'

                pass
            else:
                list_diy_h_location = get_h_location_func(input_body_list[k][0]['data'])  # 确定其他项的高度
                for l in range(len(list_diy_h_location)):
                    if abs(list_name_position_correct[i] - list_diy_h_location[l]) < judge:  # todo 如何判断高度最近
                        list_out_child.append(input_body_list[k][0]['data'][l]['text'])

                        dict_out_child[list_title[k]] = input_body_list[k][0]['data'][l]['text']

                        break
                    else:
                        pass
                if len(list_out_child) == k:
                    list_out_child.append('空')

                    dict_out_child[list_title[k]] = '空'

        list_out.append(list_out_child)

        dict_out.append(dict_out_child)

    return list_out, dict_out


def read_keywords(path):
    keys = configparser_custom()
    keys.read(path, 'UTF-8')
    keys_read = keys.items("keywords")
    return keys_read


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
            print(Back.GREEN + type)
            return type


def main_pross(cvimg, demo_or_not, hospital_lock, report_type_lock):
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
    if hospital_lock == False:
        hospital = PRE_pross.charactor_match_hospital_name(report_overview, '医院')
    if hospital_lock == True:
        hospital = '复旦大学附属华山医院'
    path_prefix = hospital
    # ocr会把间隔大的文字分开识别，大概率优先识别为中文字符

    if hospital is None:
        # print('未能识别医院信息')
        return '错误：未能识别所属医院，请拍摄完整的报告单图片，并保证纸面平整'

    # 读取报告类型关键词
    if report_type_lock is False:
        report_type = type_judge(lstKwds_need_judge=report_overview,
                                 conf_path='conf')

    if report_type_lock is True:
        report_type = '肺功能'

    path_suffix = f'-{report_type}'

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

    # 新，读配置文件为dict，方便使用配置文件的标题对数据组合进行自动标识
    dict_boxes_conf = conf.as_dict()["boxes"]
    '''
    for list_i in dict_boxes_conf.items():
        print(list_i)
    '''
    list_title = []
    for list_i in dict_boxes_conf:
        remove_digits = str.maketrans('', '', digits)
        list_i = list_i.translate(remove_digits)
        list_title.append(list_i)

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

    # print(answer[1][0]['data'][1]['text'])
    # 以下开始处理识别回传数据
    # 格式说明： ?_position[],先长后高，左上开始顺时针4点

    # 开始根据坐标对齐

    if int(len(answer)) % 2 == 1:
        print('需要对齐的列表有问题')
        return '错误:需要对齐的列表有问题'
    answer1 = []
    answer2 = []
    list_title1 = []
    list_title2 = []
    for i in range(len(answer)):
        if i < int(len(answer) / 2):
            answer1.append(answer[i])
            list_title1.append(list_title[i])
        else:
            answer2.append(answer[i])
            list_title2.append(list_title[i])
    list_direct, dict_direct = data_align_new(answer1, list_title1)
    list_direct2, dict_direct2 = data_align_new(answer2, list_title2)
    list_direct.extend(list_direct2)
    dict_direct.extend(dict_direct2)

    bloodtest_list = []
    for i in range(len(dict_direct)):
        bloodtest_single = OrderedDict()
        for j in range(len(dict_direct[i])):
            try:
                bloodtest_single[list_title[j]] = dict_direct[i][list_title[j]]
            except:
                bloodtest_single[list_title[j]] = '空'
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

    main_pross(cvimg=img_input,
               demo_or_not=1,
               hospital_lock=False,
               report_type_lock=False)
