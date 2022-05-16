#如何安装并运行此项目：
##安装
1. 下载并解压此项目

2. 此项目建议使用Python3.6，为方便管理，建议使用虚拟环境（venv（或称为pipenv）或者Anaconda皆可）

+ 虚拟环境建议优先使用pipenv，按照“Pipfile”安装所需组件。也可以使用conda，按照“requirements.txt”安装所需组件。

3. 如果使用Pycharm，应该会自动识别并自动安装对应pipenv。如果使用vscode，方便起见建议使用Anaconda并手动在使用虚拟环境的vscode中通过执行“pip install 包名”的命令来安装依赖包（参照requirements.txt依次安装）

+ OCR识别器如使用cpu则将"pip install paddlepaddle-gpu==2.2.2"替换为“pip install paddlepaddle”

4. 安装mongoDB，网址
https://www.mongodb.com/try/download/community

5. 参照下面的“运行方法”部分进行运行。

####至此，主程序可以运行基于cpu的识别存储并显示显示用户界面。

+ 本项目可以将数据库与OCR识别服务器在局域网上的其他设备上分开启动运行。

##目前所需要的运行条件（显卡计算）：

+ cuda10.2+cudnn7（paddleOCR运行所需）
+ paddlehub
+ mongoDB

* paddleHUB切换为显卡计算的方法：安装cuda和cudnn后，复制paddleHUB配置文件到C:\Users\"你的用户名"\\.paddlehub\modules\chinese_ocr_db_crnn_server
并改显卡ID，如果没有此文件夹请运行paddlehub一遍（一般台式为0，笔记本为1，根据任务管理器的独显标号为准）

##运行方法：

1. paddlehub启动：

- 如果是anaconda Prompt:

  activate [T]Blood_repo_ocr 
（切换至相应虚拟环境）

  hub serving start -m chinese_ocr_db_crnn_server -p 8866

- 如果是pipenv Prompt：

  hub serving start -m chinese_ocr_db_crnn_server -p 8866

2. 主程序启动：
+ 开发环境：运行WEB_MAIN.py
+ 部署环境：双击打开WEB_MAIN.exe即可

3. 进入用户界面：
+ 浏览器进入
localhost:8080
即可