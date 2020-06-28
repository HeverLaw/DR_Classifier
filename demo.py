# -*- coding: utf-8 -*-
import os
import torch
import base64
import re
from io import BytesIO
from config import cfg
from tools.predictor import Classifier
from PIL import Image


# 测试工具
def image_to_base64(image_path):
    img = Image.open(image_path).convert('RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


# 测试工具
def base64_to_image(base64_str, image_path=None):
    base64_str = base64_str.decode('utf-8')
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img


def load_config():
    '''
    加载配置文件
    :return: cfg配置文件对象
    '''
    print(torch.__version__)
    cfg.freeze()
    return cfg


# 配置模型路径
model_path = {
    'resnet34': './models/resnet34.pth',
}


if __name__ == '__main__':
    # 加载测试数据
    image_dir = './images'
    image_lists = os.listdir(image_dir)

    # 加载配置信息
    cfg = load_config()

    # 模型存起来，等待接收数据，收到请求调用model.run_image即可
    model = Classifier(cfg, model_path, device='cpu')

    # 加载模型
    model.load_model()

    # 模拟接收数据
    path = os.path.join(image_dir, image_lists[0])
    # 假设接收到base64的图像
    base64_image = image_to_base64(path)
    # 再转化为PIL.Image.Image对象
    img = base64_to_image(base64_image)

    import time
    # 方式一：直接传入PIL.Image.Image对象
    prediction, score = model.run_image(img=img)
    # 输出分类结果，注意分类的index是[0,5)，因此需要加1
    print('预测分类为：', prediction + 1, '分类得分为：',  score)

    # 方式二：使用路径读取图像
    prediction, score = model.run_image(path=os.path.join(image_dir, image_lists[1]))
    # 输出分类结果，注意分类的index是[0,5)，因此需要加1
    print('预测分类为：', prediction + 1, '分类得分为：',  score)
