# coding=utf-8
import os
import shutil
import MAIN_PROSS_re
import PRE_pross

if __name__ == "__main__":
    shutil.rmtree('temp')
    os.mkdir('temp')
    os.mkdir('temp/DEMO')
    os.mkdir('temp/ocr_result')
    os.mkdir('temp/DEMO/mask')

    img_orig_path = 'OCR_IMG/Input_IMG/课程3.jpg'
    img_input = PRE_pross.cv_imread_chs(img_orig_path)

    MAIN_PROSS_re.main_pross(cvimg=img_input,
                             demo_or_not=1,
                             hospital_lock=False,
                             report_type_lock=False)
