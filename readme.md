目前所需要的运行条件：

+ cuda10.2+cudnn7（paddleOCR运行所需）
+ paddlehub
+ mongoDB

paddleHUB切换为显卡计算的方法：复制paddleHUB配置文件到C:\Users\MI\.paddlehub\modules\chinese_ocr_db_crnn_mobile
并改显卡ID（根据任务管理器的独显标号为准）

运行方法：

1、paddlehub启动：

- 如果是anaconda Prompt:

  activate [T]Blood_repo_ocr

  hub serving start -m chinese_ocr_db_crnn_mobile -p 8866

- 如果是pipenv Prompt：

  hub serving start -m chinese_ocr_db_crnn_mobile -p 8866

2、主程序启动：
+ 双击打开WEB_MAIN.exe即可

3、进入用户界面：
+ 浏览器进入
localhost:8080
即可