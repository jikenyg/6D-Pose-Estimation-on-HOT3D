INPUT_SIZE = 256
CELL_SIZE = 64
MASK_SIZE = 64
USE_6D = True
N_POSE_BIN = 4608 
N_BIN_TO_KEEP = 9

WEIGHT_TRANS_CLF = 2.0
WEIGHT_DEPTH_CLF = 2.0  
WEIGHT_QUAT_CLF = 0.5
WEIGHT_MASK = 10.0
WEIGHT_QUAT_REG = 1.0
WEIGHT_TRANS_REG = 1.0
WEIGHT_DEPTH_REG = 0.2


RANDOM_SEED = 2022      # random seed
INPUT_IMG_SIZE = 256    # the input image size of network
OUTPUT_MASK_SIZE = 64   # the output mask size of network
Tz_BINS_NUM = 500      # the number of discretized depth bins
POSE_SIGMA = 0.03  # standard deviation of quaternion bin distribution 原本是0.03
DEPTH_SIGMA = 0.5     # standard deviation of depth Gaussian distribution
TRANS_SIGMA = 10 / INPUT_SIZE  # standard deviation of noise in 2D center
MODEL_UNIT_SCALE = 0.001 

DATASET_ROOT = "./data/bop_datasets"
EVAL_ROOT = "./data/bop_datasets_eval"
VOC_BG_ROOT = "./data/VOCdevkit/VOC2012"




END_LR = 5e-6
START_LR = 2e-4
END_LR_FT = 5e-7
START_LR_FT = 5e-6
DECAY_WEIGHT = 1e-3


USE_CACHE = True
CACHE_MASK = True
ZOOM_PAD_SCALE = 1.5
ZOOM_SCALE_RATIO = 0.25
ZOOM_SHIFT_RATIO = 0.25   # center shift
COLOR_AUG_PROB = 0.8
CHANGE_BG_PROB = 0.8      # the prob for changing the background



RZ_ROTATION_AUG = False

DATASET_CONFIG = {
    'hot3d': {
        'width': 1280,         # 图像宽度
        'height': 1024,        # 图像高度
        'Tz_near': 0.01,      # 最近深度
        'Tz_far': 1.17,       # 最远深度
        'num_class': 33,       # 类别数量
        'id2mod': {v: f"obj_{v:06d}" for v in range(1, 34)},  # 从1到33的对象ID映射
        'id2cls': {v: v-1 for v in range(1, 34)},  # 从对象ID到类别索引的映射
        'model_folders': {
            'train_pbr': 'models',
            'train': 'models',         # 添加这行
            'test': 'models',
            'models_eval': 'models_eval'    # 添加这行，映射 models_eval 到 models
        },
        'train_set': ['train_pbr'],
        'finetune_set': ['train'],
        'test_set': ['test'],
    },  # HOT3D
}


