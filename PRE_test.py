# coding=utf-8
import os


if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__)) # 'C:\\Users\\user'
    folder = os.path.basename(path)  # 'user'

    root_dir1 = os.path.dirname(__file__)
    root_dir2 = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir4 = os.path.dirname(os.path.abspath('.'))
    path = os.getcwd()  # 'C:\\Users\\user'

    print('1')
