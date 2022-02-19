import os
import sys
import threading
import WEB_MAIN

def ocr_core():
    os.system('hub serving start -m chinese_ocr_db_crnn_mobile -p 8866')

def user_interface_main_pross():
    WEB_MAIN.run()

thread = []
thread.append(threading.Thread(target =ocr_core))
thread.append(threading.Thread(target =user_interface_main_pross))
print(thread)
if __name__ == '__main__':
    print(os.path.abspath(os.curdir))
    print(os.getcwd())

    os.system('conda activate [T]Blood_repo_ocr_gpu')
    os.system('echo Wait till...')

    for t in thread:
        t.start()
    os.system('pause')