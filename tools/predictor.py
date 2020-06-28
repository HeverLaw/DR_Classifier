# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np
import random
from config import cfg
from network import Network
from torchvision import transforms
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torch.nn.functional import softmax


# Please specify the ID of graphics cards that you want to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Resize(object):
    def __init__(self, min_size, max_size):
        '''
        数据转换的工具，用于保持比例resize
        :param min_size:
        :param max_size:
        '''
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        # image = F.resize(image, (512,512))
        image = F.resize(image, size)
        return image


def make_transformer(cfg):
    '''
    创造一个transformer，用于数据预处理
    :param cfg:
    :return:
    '''
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    return transformer


class Classifier:
    def __init__(self, config, model_path, device='cpu'):
        '''
        使用resnet34或resnet18，直接使用库里面的transforms
        :param config: config对象
        :param model_path: 模型的路径
        :param device: 使用cpu或者cuda
        '''
        super(Classifier, self).__init__()
        self.cfg = config
        self.model = None
        self.transformer = make_transformer(cfg)
        self.model_path = model_path[cfg.BACKBONE]
        self.device = torch.device(device)

    def load_model(self):
        # TODO:填写以项目为根目录的模型路径
        self.model = Network(cfg)
        # 读取模型权重
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        # 开启测试模式
        self.model.eval()

    def run_image(self, path=None, img=None, is_test=False):
        # 接收图像
        assert path is not None or img is not None, '至少传入path或者PIL.Image格式的图像'
        if path is not None:
            img = Image.open(path).convert("RGB")
            img = self.pre_process_image(img)
        else:
            assert isinstance(img, PIL.Image.Image), '输入图像需要为Image.Image的RGB格式'
            img = self.pre_process_image(img)
        ##############################################
        with torch.no_grad():
            img = img.to(self.device)
            feature, prediction = self.model(img)
            score_list = softmax(prediction, dim=1).cpu().numpy()
            prediction = torch.argmax(prediction, dim=1).item()
        if is_test:
            return prediction, score_list
        else:
            return prediction, score_list[0][prediction]


    def pre_process_image(self, img):
        '''
        输入图像需要为
        :param path:
        :return:
        '''
        img = np.array(img)
        img = self.transformer(img)
        # 只跑一张图像，添加第0维度（网络只支持4维的）
        img = torch.unsqueeze(img, 0)
        return img


