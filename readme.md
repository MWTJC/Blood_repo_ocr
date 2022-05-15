paddleHUB切换为显卡：复制配置文件到C:\Users\MI\.paddlehub\modules\chinese_ocr_db_crnn_mobile
并改显卡ID

运行要求：paddlehub启动：

- anaconda Prompt:

  activate [T]Blood_repo_ocr

  hub serving start -m chinese_ocr_db_crnn_mobile -p 8866

- pipenv Prompt：

  hub serving start -m chinese_ocr_db_crnn_mobile -p 8866

mongoDB启动