import os

from yacs.config import CfgNode as CN

_C = CN()
# _C.NAME = 'sgd000105_trans_adam'
# _C.NAME = 'sgd000105_trans'
# _C.NAME = 'sgd000105no_trans'

# _C.NAME = 'model2_no_trans'
# _C.NAME = 'model2_no_trans_focal'
_C.NAME = 'other_focal01_to1_pool'
_C.DESCRIPTION = ''
# _C.NAME = 'model1_focal01_to1_pool'
_C.NAME = 'pretrain_focal01_to1_pool'
_C.NAME = 'pretrain_fo_pool'
_C.PRETRAINED = True
_C.PRETRAIN_MODEL = './pretrained_model/pretrain_32000_freeze.pth'
_C.FOCAL_LOSS_ALPHA = [0.1, 5, 2, 2, 2]
# _C.NAME = 'first'
_C.OUTPUT_DIR = os.path.join('./output', _C.NAME)
_C.DEVICE = 'cuda'
_C.BACKBONE = 'resnet34'
# 这个不能变，如果变需要重新定义模型
_C.NUM_FEATURES = 1000
_C.NUM_CATEGORY = 5
_C.FREEZE_CONV_BODY_AT = 2


_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (448, )  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 512
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 448
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 512
# Values to be used for image normalization
# _C.INPUT.PIXEL_MEAN = [105.0051/255, 73.4359/255, 52.3038/255]
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# _C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False
# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.00
_C.INPUT.CONTRAST = 0.00
_C.INPUT.SATURATION = 0.00
_C.INPUT.HUE = 0.00

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = 'train'
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = 'test'


# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# pretrain
_C.SOLVER.EPOCH = 30
# 正常训练
# _C.SOLVER.EPOCH = 30
# _C.SOLVER.MAX_ITER = 15000
_C.SOLVER.BASE_LR = 0.005
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.2
# 这个数字比较难定，会和IMS_PER_BATCH有关系
# _C.SOLVER.STEPS = (2000, 5000)
_C.SOLVER.STEPS = (3000, 7000, 11000)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

# _C.SOLVER.CHECKPOINT_PERIOD = 10000

_C.SOLVER.IMS_PER_BATCH = 16




