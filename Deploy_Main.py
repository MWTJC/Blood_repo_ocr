import os
import threading

def ocr_core():
    os.system('hub serving start -m chinese_ocr_db_crnn_mobile -p 8866')

def user_interface_main_pross():
    os.system('python WEB_MAIN.py')

thread = []
thread.append(threading.Thread(target =ocr_core))
thread.append(threading.Thread(target =user_interface_main_pross))
print(thread)
if __name__ == '__main__':
    for t in thread:
        t.start()