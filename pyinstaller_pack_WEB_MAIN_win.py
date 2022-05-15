from PyInstaller.__main__ import run

if __name__ == '__main__':
    opts = ['WEB_MAIN.py',  # 主程序文件
            # '-n flask',  # 可执行文件名称
            # '-F',  # 打包单文件
            # '-w', #是否以控制台黑窗口运行
            # r'--icon=E:/图标/leaves_16px_1218386_easyicon.net.ico',  # 可执行程序图标
            '-y',
            '--clean',  # 清除缓存
            '--win-private-assemblies',  # 私人组件把跟这个程序捆绑的共享的组件都改为私有的
            # '--workpath=build',
            # '--add-data=templates;templates',  # 打包包含的html页面
            '--add-data=static;static',  # 打包包含的静态资源
            '--add-data=conf;conf',  # 打包
            '--add-data=OCR_IMG;OCR_IMG',  # 打包
            '--add-data=C:/Users/TJC/.paddlehub/modules/chinese_ocr_db_crnn_mobile/module.py;conf',
            # '--distpath=build', # 最终位置
            '--specpath=./'
            ]

    run(opts)

