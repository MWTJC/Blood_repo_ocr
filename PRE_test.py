from PIL import Image
import cv2
import numpy
import PRE_pross


if __name__ == "__main__":
    img_org = cv2.imread('DEMO/mask/report_repo_num1.jpg')
    PRE_pross.image_border(img_org, 'R.jpg')