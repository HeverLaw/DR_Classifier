## 安装

### 环境

python=3.7.7

pytorch >= 1.0

yacs

### 安装步骤

```
conda create -n classifier python=3.7.7
conda activate classifier
# 可以使用清华镜像
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch torchvision cudatoolkit=10.0
# CPU版本
# conda install pytorch torchvision cpuonly
pip install yacs
```

## 保存模型的文件

模型文件存放路径如字典所示，使用哪个模型由配置文件中BACKBONE决定，默认使用resnet34

```
model_path = {
    'resnet34': './models/resnet34.pth',
}
```

## 模型性能说明

单张图像的处理时间（含IO和数据预处理）：gpu（RTX-2080）约0.04s/张，cpu（i7-8700）约0.22s/张

### Resnet34模型性能（验证集）

|           | level1 | level2 | level3 | level4 | level5 |
| --------- | ------ | ------ | ------ | ------ | ------ |
| 准确率(%) | 96.59  | 16.45  | 65.68  | 35.53  | 58.18  |

平均准确率：84.18%

ROC-AUC

|      | level1 | level2 | level3 | level4 | level5 | 微平均AUC | 宏平均AUC |
| ---- | ------ | ------ | ------ | ------ | ------ | --------- | --------- |
| AUC  | 0.8911 | 0.7244 | 0.9012 | 0.9721 | 0.9843 | 0.9658    | 0.8950    |

## 模型使用说明

参考./demo.py

```
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
```

## 图像预处理

图像的预处理将在读取图像后进行，集成在Classifier.pre_process_image(self, img)中，Classifier可以通过两种方式接收图像

### 方式一：直接传入PIL.Image.Image对象

```
prediction, score = model.run_image(img=img)
```

若传入图像是base64，可以使用以下接口先转化为PIL.Image.Image对象，再传送到run_image接口。

```
img = base64_to_image(base64_image)
```

### 方式二：使用文件路径方式读取图像

```
prediction, score = model.run_image(path=os.path.join(image_dir, image_lists[1]))
```



