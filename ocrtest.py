# coding=utf-8
import MAIN_PROSS
import cv2
from pprint import pprint

if __name__ == '__main__':
    img = cv2.imread('OCR_IMG\\Input_IMG\\zs-blood3.jpg')
    pre_response = MAIN_PROSS.net_OCR(img, ip)
    if pre_response is 'OCROFFLINE':
        print ('错误：OCR离线')
    elif len(pre_response.json()["results"]) == 0:
        pprint (f"错误：OCR没有正常工作,{pre_response.text}")
    else:
        pprint(pre_response.json()["results"][0]["data"])