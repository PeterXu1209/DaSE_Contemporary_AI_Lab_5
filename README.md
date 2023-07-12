## Multimodal Sentiment Big Homework  多模态情感分析大作业

本仓库包含了当代人工智能的第五次大作业的代码

学号：10205501440

名字：徐沛楹

## Setup 开始

This implemetation is based on Python3. To run the code, you need the following dependencies:

- chardet==4.0.0
- numpy==1.23.5
- Pillow==9.3.0
- Pillow==9.4.0
- Pillow==10.0.0
- scikit_learn==1.2.1
- torch==2.0.0+cu118
- torch==2.0.0
- torchvision==0.15.0
- torchvision==0.15.1+cu118
- tqdm==4.65.0
- transformers==4.28.0

You can simply run 你可以直接运行下面的脚本

```
pip install -r requirements.txt
```

## Repository structure 仓库结构


```
|-- Multimodal-Sentiment-Analysis
    |-- bert-base-multilingual-cased # 预训练模型 （包含其内容的文件详见网盘链接）
    |-- bert-base-uncased # 预训练模型 （包含其内容的文件详见网盘链接）
    |-- README.md # 本文件
    |-- requirements.txt
    |-- Trainer.py # 包含主函数的所用到的Trainer
    |-- data
    |   |-- test.json # 导出文件
    |   |-- test_without_label.txt # 源文件
    |   |-- train.json # 导出文件
    |   |-- train.txt # 源文件
    |   |-- data # 原始数据 详见网盘链接
    |-- data_collected # 实验采集到的数据
    |-- Models
    |   |-- __init__.py
    |   |-- cat.py # 使用cat方法融合的模型
    |   |-- default.py # 使用default方法融合的模型
    |   |-- ImageModel.py # 图像模型
    |   |-- TextModel.py # 文本模型
    |-- output # 存放了训练完毕的模型
    |-- common.py # 存放了有关原始数据处理的函数
    |-- config.py # 存放了训练的默认参数
    |-- data_processor.py # 存放了有关训练时数据处理的函数
    |-- main.py # 主函数
    |-- output.txt # 预测结果
    |-- README.md # 本文件
    |-- requirements.txt
    |-- Trainer.py # 训练器
    |-- 实验报告 # 详见邮件附件
    

```

## Run and Predict 运行与预测

想要运行本作业的代码，请先训练模型，训练完毕模型后请记住训练得到的模型的路径，将do_train改为do_test，再配合其他参数即可转为预测模式

训练模型

```shell
python main.py --do_train --epoch 5 --text_pretrained_model bert-base-uncased --fuse_model_type cat
```

若要训练单模态模型，请在后面加上--text_only 或 --img_only

预测模型

```shell
python main.py --do_test --text_pretrained_model bert-base-uncased --fuse_model_type cat --load_model_path $模型目录$ 
```

所有可以调节的参数如下所示

![image-20230711235211314](D:\Code\Python_projects\ComtemporaryAI\project5\assets\image-20230711235211314.png)

通过下一节的完整代码下载处下载了完整了完整的代码与文件后，若你在文本预训练模型中选择了bert-base-uncased或bert-base-multilingual-cased，则即可直接无需等待地开始训练，因为文件夹中包含了完整地文件。若选择了其他模型则需要等待预训练模型下载完成。

## Place To Find Full Files 完整代码下载处

由于大小限制，GitHub的文件夹中不包含预训练模型与data，想要获得完整的文件请通过下面的网盘链接下载：



## Whole Project Report 完整报告与实验结果

完整的报告与实验结果详情请见邮件的附件的压缩包中随附的实验报告
