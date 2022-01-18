import MAIN_PROSS
import cv2
from pprint import pprint

if __name__ == '__main__':
    img = cv2.imread('R.jpg')
    res = MAIN_PROSS.net_OCR(img)
    pprint(res.json()["results"][0]["data"])